#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_URL = process.env.BANGLA_WHISTLE_URL || '';
const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);

function usage() {
  console.log(`BanglaWhistle client

Usage:
  npx bangla-whistle transcribe <audio-file> [--remote] [--url URL]
  npx bangla-whistle serve [--image IMAGE]

Local transcription uses Python + PyTorch and a checkout containing the model.
Use --remote to call the hosted API; --url overrides its default address.
`);
}

function optionValue(name) {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`${name} requires a value`);
  args.splice(index, 2);
  return value;
}

function hasOption(name) {
  const index = args.indexOf(name);
  if (index < 0) return false;
  args.splice(index, 1);
  return true;
}

function pythonCommand() {
  return process.env.PYTHON || 'python3';
}

function pythonHasTorch() {
  const check = spawnSync(pythonCommand(), ['-c', 'import torch'], { stdio: 'ignore' });
  return check.status === 0;
}

function findCheckout() {
  const roots = [process.env.BANGLA_WHISTLE_ROOT, process.cwd(), packageDir].filter(Boolean);
  for (const start of roots) {
    let dir = path.resolve(start);
    while (true) {
      if (existsSync(path.join(dir, 'runtime', 'transcribe.py'))) {
        const model = process.env.BANGLA_WHISTLE_MODEL || [
          path.join(dir, 'edge-int8'),
          path.join(dir, 'checkpoints', 'rslt3'),
        ].find((candidate) => existsSync(path.join(candidate, 'student.int8.pt')) || existsSync(path.join(candidate, 'student.pt')));
        if (!model) return undefined;
        return { root: dir, model };
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  return undefined;
}

async function remoteTranscribe(audioPath, url) {
  const data = await readFile(audioPath);
  const form = new FormData();
  form.append('file', new Blob([data]), path.basename(audioPath));
  const response = await fetch(`${url.replace(/\/$/, '')}/v1/transcribe`, { method: 'POST', body: form });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.detail || `Request failed with HTTP ${response.status}`);
  process.stdout.write(`${result.text ?? ''}\n`);
}

function runLocal(checkout, audioPath) {
  const script = path.join(checkout.root, 'runtime', 'transcribe.py');
  const result = spawnSync(pythonCommand(), [script, '--model', checkout.model, '--audio', audioPath], {
    cwd: checkout.root,
    encoding: 'utf8',
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.status !== 0) process.exitCode = result.status ?? 1;
}

function serve(image) {
  const checkout = findCheckout();
  if (checkout && pythonHasTorch()) {
    const check = spawnSync(pythonCommand(), ['-c', 'import fastapi, torch, transformers, uvicorn'], { stdio: 'ignore' });
    if (check.status === 0) {
      const child = spawn(pythonCommand(), ['-m', 'uvicorn', 'app:app', '--app-dir', 'site', '--host', '127.0.0.1', '--port', '8080', '--workers', '1'], {
        cwd: checkout.root,
        stdio: 'inherit',
      });
      child.on('exit', (code) => { process.exitCode = code ?? 1; });
      return;
    }
  }
  const child = spawn('docker', ['run', '--rm', '-p', '8080:8080', image], { stdio: 'inherit' });
  child.on('error', () => console.error(`Install Docker or run from a checkout with Python, PyTorch, and Uvicorn. Build the server image as ${image} first.`));
  child.on('exit', (code) => { process.exitCode = code ?? 1; });
}

async function main() {
  const command = args.shift();
  if (!command || command === '--help' || command === '-h') return usage();
  if (command === 'transcribe') {
    const audioPath = args.shift();
    const url = optionValue('--url');
    const remote = hasOption('--remote') || Boolean(url);
    if (!audioPath || args.length) throw new Error('Provide one audio file and supported options.');
    if (remote) {
      const target = url || DEFAULT_URL;
      if (!target) throw new Error('Pass --url https://your-space.hf.space or set BANGLA_WHISTLE_URL.');
      return remoteTranscribe(audioPath, target);
    }
    if (pythonHasTorch()) {
      const checkout = findCheckout();
      if (checkout) return runLocal(checkout, audioPath);
      throw new Error('Python can import PyTorch, but no BanglaWhistle checkout was found. Set BANGLA_WHISTLE_ROOT and BANGLA_WHISTLE_MODEL, or use --remote.');
    }
    throw new Error('Python/PyTorch is unavailable. Install PyTorch for local inference, run the Docker image, or use --remote.');
  }
  if (command === 'serve') {
    const image = optionValue('--image') || 'bangla-whistle';
    if (args.length) throw new Error('Unsupported serve option.');
    return serve(image);
  }
  throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
  console.error(`bangla-whistle: ${error.message}`);
  process.exitCode = 1;
});
