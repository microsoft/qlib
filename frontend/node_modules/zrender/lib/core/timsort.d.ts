declare type CompareFunc<T> = (a: T, b: T) => number;
export default function sort<T>(array: T[], compare: CompareFunc<T>, lo?: number, hi?: number): void;
export {};
