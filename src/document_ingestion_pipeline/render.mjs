// stdin: JSON array of block lists. stdout: JSON array of markdown strings, one per list.
// kordoc 4.24.11 exports blocksToMarkdown but keeps toHtmlTables in a hashed internal chunk,
// so the chunk is located by its export list. Bumping kordoc means re-checking this lookup.
import { readdir, readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire(join(process.cwd(), "package.json"));
const dist = dirname(require.resolve("kordoc"));
const kordoc = await import(pathToFileURL(join(dist, "index.js")).href);

async function findInternal(name) {
  for (const file of await readdir(dist)) {
    if (!file.startsWith("chunk-") || !file.endsWith(".js")) continue;
    const source = await readFile(join(dist, file), "utf8");
    const exports = source.slice(source.lastIndexOf("export {"));
    if (new RegExp(`\\b${name}\\b`).test(exports)) {
      const mod = await import(pathToFileURL(join(dist, file)).href);
      if (typeof mod[name] === "function") return mod[name];
    }
  }
  throw new Error(`kordoc ${kordoc.VERSION}: internal ${name} not found`);
}

const toHtmlTables = await findInternal("toHtmlTables");

let input = "";
for await (const chunk of process.stdin) input += chunk;
const batches = JSON.parse(input);
const out = batches.map((blocks) => toHtmlTables(kordoc.blocksToMarkdown(blocks)));
process.stdout.write(JSON.stringify(out));
