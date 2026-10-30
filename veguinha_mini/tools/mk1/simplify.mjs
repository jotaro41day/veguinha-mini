import fs from 'node:fs';
import { MeshoptSimplifier } from 'meshoptimizer';

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error('Usage: node simplify.mjs input.json output.json');
}

await MeshoptSimplifier.ready;
const groups = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const output = {};
for (const group of groups) {
  const indices = new Uint32Array(group.indices);
  const vertices = new Float32Array(group.vertices);
  const [simplified, error] = MeshoptSimplifier.simplify(
    indices,
    vertices,
    3,
    group.target * 3,
    1.0,
  );
  output[group.key] = { indices: Array.from(simplified), error };
}
fs.writeFileSync(outputPath, JSON.stringify(output));
