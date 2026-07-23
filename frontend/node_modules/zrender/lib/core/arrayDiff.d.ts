declare type EqualFunc<T> = (a: T, b: T) => boolean;
declare type DiffComponent = {
    count: number;
    added: boolean;
    removed: boolean;
    indices: number[];
};
export default function arrayDiff<T>(oldArr: T[], newArr: T[], equal?: EqualFunc<T>): DiffComponent[];
export {};
