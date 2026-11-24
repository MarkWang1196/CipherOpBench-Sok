// Numeric simulation of CKKS slot values as [real, imaginary] pairs.
export function multiplyComplex(left, right) {
  const [a, b] = left;
  const [c, d] = right;
  return [a * c - b * d, a * d + b * c];
}

export function sumComplexVectors(vectors) {
  return vectors[0].map((_, i) => vectors.reduce(
    ([real, imag], vector) => [real + vector[i][0], imag + vector[i][1]], [0, 0]
  ));
}

export function complexLabel([real, imag]) {
  return `${real}${imag < 0 ? "−" : "+"}${Math.abs(imag)}i`;
}
