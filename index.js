require('dotenv').config();
const { Client, LocalAuth } = require('whatsapp-web.js');
const fs = require('fs');
const qrcode = require('qrcode-terminal');
const path = require('path');
const https = require('https');
const http = require('http');
const { spawn } = require('child_process');

const WHATSAPP_OWNER_ID = process.env.WHATSAPP_OWNER_ID;
let WHATSAPP_OWNER_NAME = process.env.WHATSAPP_OWNER_NAME || "the owner";
const CACHE_FILE = 'chat_cache.json';
const LLAMA_PORT = 18642;
const LLM_MODEL_SIZE = process.env.LLM_MODEL_SIZE || 'E4B';
const GPU_LAYERS = process.env.GPU_LAYERS || '99';

const MODELS_CONFIG = {
    'E4B': {
        repo: 'bartowski/google_gemma-4-E4B-it-GGUF',
        model: 'google_gemma-4-E4B-it-Q4_K_M.gguf',
        mmproj: 'mmproj-google_gemma-4-E4B-it-f16.gguf'
    },
    '26B': {
        repo: 'unsloth/gemma-4-26B-A4B-it-GGUF',
        model: 'gemma-4-26B-A4B-it-UD-IQ4_NL.gguf',
        mmproj: 'mmproj-F16.gguf'
    }
};

const getCleanNumber = (id) => {
    if (!id) return '';
    return id.split('@')[0].split(':')[0];
};

// ─── File Downloader ────────────────────────────────────────────────────────

async function downloadFile(url, dest) {
    if (fs.existsSync(dest) && fs.statSync(dest).size > 0) return;
    if (fs.existsSync(dest)) fs.unlinkSync(dest);
    const dir = path.dirname(dest);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    console.log(`\nDownloading: ${path.basename(dest)}`);
    return new Promise((resolve, reject) => {
        const request = (sourceUrl) => {
            const options = {};
            if (process.env.HF_TOKEN && sourceUrl.includes('huggingface.co')) {
                options.headers = { 'Authorization': `Bearer ${process.env.HF_TOKEN}` };
            }
            https.get(sourceUrl, options, (response) => {
                if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                    return request(new URL(response.headers.location, sourceUrl).href);
                }
                if (response.statusCode !== 200) return reject(new Error(`HTTP ${response.statusCode} for ${sourceUrl}`));
                const file = fs.createWriteStream(dest);
                let downloaded = 0;
                const total = parseInt(response.headers['content-length'] || 0, 10);
                response.on('data', chunk => {
                    downloaded += chunk.length;
                    if (total) process.stdout.write(`\r  ${Math.floor(downloaded / total * 100)}% (${(downloaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB)`);
                });
                response.pipe(file);
                file.on('finish', () => { file.close(); process.stdout.write('\n'); resolve(); });
                file.on('error', err => { fs.unlink(dest, () => { }); reject(err); });
            }).on('error', reject);
        };
        request(url);
    });
}

// ─── llama-server Helpers ───────────────────────────────────────────────────

async function ensureLlamaServer() {
    const release = 'b8642';
    const serverPath = path.join(__dirname, 'models', `llama-server-${release}.exe`);
    if (fs.existsSync(serverPath) && fs.statSync(serverPath).size > 0) return serverPath;

    // Download Server Binary
    const binUrl = `https://github.com/ggml-org/llama.cpp/releases/download/${release}/llama-${release}-bin-win-cuda-12.4-x64.zip`;
    const binZip = path.join(__dirname, 'models', 'llama-server.zip');

    // Download CUDA Runtime (DLLs)
    const dllUrl = `https://github.com/ggml-org/llama.cpp/releases/download/${release}/cudart-llama-bin-win-cuda-12.4-x64.zip`;
    const dllZip = path.join(__dirname, 'models', 'cudart.zip');

    console.log('Fetching llama-server.exe and CUDA dependencies...');
    await downloadFile(binUrl, binZip);
    await downloadFile(dllUrl, dllZip);

    const extractToModels = async (zip, name) => {
        const tempDir = path.join(__dirname, 'models', '_temp_' + name);
        if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

        await new Promise((resolve, reject) => {
            const ps = spawn('powershell', [
                '-Command',
                `Expand-Archive -Path '${zip}' -DestinationPath '${tempDir}' -Force`
            ]);
            ps.on('close', code => code === 0 ? resolve() : reject(new Error(`Unzip ${name} failed`)));
        });

        const copyRecursive = (src, dest) => {
            const items = fs.readdirSync(src);
            for (const item of items) {
                const s = path.join(src, item);
                const d = path.join(dest, item);
                if (fs.statSync(s).isDirectory()) {
                    if (!fs.existsSync(d)) fs.mkdirSync(d);
                    copyRecursive(s, d);
                } else {
                    fs.copyFileSync(s, d);
                }
            }
        };
        copyRecursive(tempDir, path.join(__dirname, 'models'));
        fs.rmSync(tempDir, { recursive: true, force: true });
        fs.unlinkSync(zip);
    };

    await extractToModels(binZip, 'server');
    await extractToModels(dllZip, 'dlls');

    // Rename to versioned binary for update tracking
    const extractedServer = path.join(__dirname, 'models', 'llama-server.exe');
    if (fs.existsSync(extractedServer)) {
        fs.renameSync(extractedServer, serverPath);
    }

    console.log('llama-server.exe and DLLs ready.');
    return serverPath;
}

async function ensureGemma4Models() {
    const config = MODELS_CONFIG[LLM_MODEL_SIZE] || MODELS_CONFIG['E4B'];
    const modelDir = path.join(__dirname, 'models');
    if (!fs.existsSync(modelDir)) fs.mkdirSync(modelDir, { recursive: true });

    const modelName = config.model;
    const mmprojName = config.mmproj;

    const modelPath = path.join(modelDir, modelName);
    const mmprojPath = path.join(modelDir, mmprojName);

    const baseUrl = `https://huggingface.co/${config.repo}/resolve/main/`;

    if (!fs.existsSync(modelPath) || fs.statSync(modelPath).size < 1000000) {
        console.log(`Downloading Gemma 4 ${LLM_MODEL_SIZE} Model: ${modelName}...`);
        await downloadFile(baseUrl + modelName, modelPath);
    }

    if (!fs.existsSync(mmprojPath) || fs.statSync(mmprojPath).size < 1000000) {
        console.log(`Downloading Gemma 4 ${LLM_MODEL_SIZE} Vision Projector: ${mmprojName}...`);
        await downloadFile(baseUrl + mmprojName, mmprojPath);
    }

    return { modelPath, mmprojPath };
}

async function waitForServer(port, timeoutMs = 60000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        try {
            await new Promise((resolve, reject) => {
                http.get(`http://127.0.0.1:${port}/health`, res => {
                    let body = '';
                    res.on('data', d => body += d);
                    res.on('end', () => {
                        try {
                            const json = JSON.parse(body);
                            if (json.status === 'ok') resolve(); else reject();
                        } catch { reject(); }
                    });
                }).on('error', reject);
            });
            return true;
        } catch {
            await new Promise(r => setTimeout(r, 500));
        }
    }
    throw new Error('llama-server failed to start within timeout');
}

async function callLlamaServer(messages) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify({ model: 'local', messages, max_tokens: 2048, stream: false });
        const req = http.request({
            hostname: '127.0.0.1',
            port: LLAMA_PORT,
            path: '/v1/chat/completions',
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
        }, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    resolve(json.choices?.[0]?.message?.content || '(no response)');
                } catch { reject(new Error('Invalid server response: ' + data.slice(0, 200))); }
            });
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}

// ─── WhatsApp Client ────────────────────────────────────────────────────────

const client = new Client({ authStrategy: new LocalAuth() });

client.on('qr', (qr) => qrcode.generate(qr, { small: true }));

client.on('ready', () => {
    console.log('WhatsApp Bot is ready!');
    if (client.info?.pushname && !process.env.WHATSAPP_OWNER_NAME) {
        WHATSAPP_OWNER_NAME = client.info.pushname;
        console.log(`Owner name: ${WHATSAPP_OWNER_NAME}`);
    }
});

client.on('message_create', async (msg) => {
    const senderId = msg.author || msg.from;
    const isOwner = (senderId === WHATSAPP_OWNER_ID) || msg.fromMe;

    const isSummary = msg.body.startsWith('!summary');
    const isHourly = msg.body.startsWith('!hourly');
    const isAsk = msg.body.startsWith('!ask');
    const isLLM = msg.body.startsWith('!llm');

    if (!(isSummary || isHourly || isAsk || isLLM)) return;
    if (!isOwner) return;

    let serverProcess = null;
    try {
        let instructionsBody = '';
        if (isSummary) instructionsBody = msg.body.slice('!summary'.length).trim();
        else if (isHourly) instructionsBody = msg.body.slice('!hourly'.length).trim();
        else if (isAsk) instructionsBody = msg.body.slice('!ask'.length).trim();
        else if (isLLM) instructionsBody = msg.body.slice('!llm'.length).trim();

        await msg.reply(`⏳ Processing request...`);

        // ── Capture quoted context & image ──────────────────────────────
        let quotedContext = '';
        let commandImage = null; // { base64: string, mimetype: string }

        if (msg.hasQuotedMsg) {
            const q = msg._data?.quotedMsg;
            if (q?.type === 'image' && q?.body?.length > 500) {
                console.log('Image captured from quoted metadata.');
                commandImage = { base64: q.body, mimetype: q.mimetype || 'image/jpeg' };
                if (q.caption) quotedContext = `[Quoted image caption]: "${q.caption}"\n\n`;
            } else {
                try {
                    const quoted = await msg.getQuotedMessage();
                    if (quoted?.hasMedia) {
                        const media = await quoted.downloadMedia();
                        if (media?.mimetype.startsWith('image/')) {
                            commandImage = { base64: media.data, mimetype: media.mimetype };
                        }
                    }
                    if (quoted?.body) quotedContext = `[Replying to]: "${quoted.body}"\n\n`;
                } catch (e) { }
            }
        } else if (msg.hasMedia) {
            const media = await msg.downloadMedia();
            if (media?.mimetype.startsWith('image/')) {
                commandImage = { base64: media.data, mimetype: media.mimetype };
            }
        }

        const instructions = quotedContext + instructionsBody;

        // ── Cache sync ─────────────────────────────────────────────────
        const midnight = new Date();
        midnight.setHours(0, 0, 0, 0);
        const midnightTS = Math.floor(midnight.getTime() / 1000);
        const timeFilterTS = isHourly ? Math.floor((Date.now() - 3600000) / 1000) : midnightTS;

        let cache = { lastSyncTimestamp: 0, messages: [] };
        if (fs.existsSync(CACHE_FILE)) {
            try {
                cache = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
                cache.messages = cache.messages.filter(m => m.timestamp >= midnightTS);
            } catch (e) { }
        }

        const chats = await client.getChats();
        let newMax = cache.lastSyncTimestamp;
        for (const chat of chats) {
            const messages = await chat.fetchMessages({ limit: 300 });
            for (const m of messages) {
                if (m.timestamp > cache.lastSyncTimestamp && !m.isStatus) {
                    const rawId = m.author || m.from || '';
                    const sid = getCleanNumber(rawId);
                    const isBot = m.fromMe === true;
                    const isO = isBot || (rawId === WHATSAPP_OWNER_ID);
                    const isBotMsg = m.body && (m.body.startsWith('!') || m.body.startsWith('⏳') || m.body.startsWith('🤖'));
                    if (!isBotMsg && m.body) {
                        cache.messages.push({ chatName: chat.name || 'Private', timestamp: m.timestamp, senderName: sid, isOwner: isO, body: m.body });
                        if (m.timestamp > newMax) newMax = m.timestamp;
                    }
                }
            }
        }
        cache.lastSyncTimestamp = newMax;
        fs.writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 2));

        // ── Spawn llama-server ─────────────────────────────────────────
        // ── Ensure Models & llama-server ───────────────────────────────
        const { modelPath, mmprojPath } = await ensureGemma4Models();
        const serverBin = await ensureLlamaServer();

        const serverArgs = [
            '-m', modelPath,
            '--mmproj', mmprojPath,
            '-ngl', GPU_LAYERS,          // offload layers to GPU
            '--port', String(LLAMA_PORT),
            '--ctx-size', '16384',
            '--no-mmap',
            '-np', '1',
            '--host', '127.0.0.1',
        ];

        console.log(`Spawning llama-server on port ${LLAMA_PORT}...`);
        serverProcess = spawn(serverBin, serverArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
        serverProcess.stdout.on('data', d => process.stdout.write(`[server] ${d}`));
        serverProcess.stderr.on('data', d => process.stderr.write(`[server] ${d}`));

        await waitForServer(LLAMA_PORT, 120000);
        console.log('Server ready. Sending request...');

        // ── Build messages array ───────────────────────────────────────
        const systemPersona = isLLM
            ? `You are a helpful assistant talking directly to ${WHATSAPP_OWNER_NAME}.`
            : `You are a helpful assistant analyzing WhatsApp chat logs for ${WHATSAPP_OWNER_NAME}. Messages tagged (YOU) were sent by them.`;

        let userContent;
        if (commandImage) {
            // Multimodal: text + image content array
            const textPart = isLLM
                ? (instructions || 'Describe this image.')
                : buildChatLogText(cache.messages, timeFilterTS, instructions);

            userContent = [
                { type: 'image_url', image_url: { url: `data:${commandImage.mimetype};base64,${commandImage.base64}` } },
                { type: 'text', text: textPart }
            ];
        } else {
            userContent = isLLM
                ? (instructions || 'Hello!')
                : buildChatLogText(cache.messages, timeFilterTS, instructions);
        }

        const llamaMessages = [
            { role: 'system', content: systemPersona },
            { role: 'user', content: userContent }
        ];

        console.time('LLM Generation Time');
        const finalResponse = await callLlamaServer(llamaMessages);
        console.timeEnd('LLM Generation Time');
        console.log('\n--- Response ---\n', finalResponse);

        await msg.reply(`🤖Whatcoms:\n\n${finalResponse}`);

    } catch (error) {
        console.error('Error:', error);
        await msg.reply('An error occurred: ' + error.message);
    } finally {
        if (serverProcess) {
            console.log('Killing llama-server...');
            serverProcess.kill('SIGTERM');
            serverProcess = null;
        }
    }
});

function buildChatLogText(messages, timeFilterTS, instructions) {
    const targets = messages.filter(m => m.timestamp >= timeFilterTS);
    let text = 'Here are the WhatsApp chat logs:\n\n';
    for (const m of targets) {
        const tag = m.isOwner ? `(YOU) ${WHATSAPP_OWNER_NAME}` : m.senderName;
        text += `[${m.chatName}] ${tag}: ${m.body}\n`;
    }
    text += `\nTASK: ${instructions || 'Summarize the key discussions above.'}`;
    return text;
}

client.initialize();