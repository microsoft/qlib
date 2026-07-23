export declare class Entry<T> {
    value: T;
    key: string | number;
    next: Entry<T>;
    prev: Entry<T>;
    constructor(val: T);
}
export declare class LinkedList<T> {
    head: Entry<T>;
    tail: Entry<T>;
    private _len;
    insert(val: T): Entry<T>;
    insertEntry(entry: Entry<T>): void;
    remove(entry: Entry<T>): void;
    len(): number;
    clear(): void;
}
export default class LRU<T> {
    private _list;
    private _maxSize;
    private _lastRemovedEntry;
    private _map;
    constructor(maxSize: number);
    put(key: string | number, value: T): T;
    get(key: string | number): T;
    clear(): void;
    len(): number;
}
