const modules = [
  '@tensorflow/tfjs-node-gpu',
  '@tensorflow/tfjs-node',
  '@tensorflow/tfjs',
];

export function loadTensorflow(): typeof import('@tensorflow/tfjs') {
  for (const module of modules) {
    try {
      return require(module);
    } catch {}
  }

  throw new Error(
    'TensorFlow.js could not be loaded. Please install one of the following packages: ' +
      modules.join(', ')
  );
}
