import { Dictionary } from './types';

// Simple LRU cache use doubly linked list
// @module zrender/core/LRU

export class Entry<T> {

    value: T

    key: string | number

    next: Entry<T>

    prev: Entry<T>

    constructor(val: T) {
        this.value = val;
    }
}
/**
 * Simple double linked list. Compared with array, it has O(1) remove operation.
 * @constructor
 */
export class LinkedList<T> {

    head: Entry<T>
    tail: Entry<T>

    private _len = 0

    /**
     * Insert a new value at the tail
     */
    insert(val: T): Entry<T> {
        const entry = new Entry(val);
        this.insertEntry(entry);
        return entry;
    }

    /**
     * Insert an entry at the tail
     */
    insertEntry(entry: Entry<T>) {
        if (!this.head) {
            this.head = this.tail = entry;
        }
        else {
            this.tail.next = entry;
            entry.prev = this.tail;
            entry.next = null;
            this.tail = entry;
        }
        this._len++;
    }

    /**
     * Remove entry.
     */
    remove(entry: Entry<T>) {
        const prev = entry.prev;
        const next = entry.next;
        if (prev) {
            prev.next = next;
        }
        else {
            // Is head
            this.head = next;
        }
        if (next) {
            next.prev = prev;
        }
        else {
            // Is tail
            this.tail = prev;
        }
        entry.next = entry.prev = null;
        this._len--;
    }

    /**
     * Get length
     */
    len(): number {
        return this._len;
    }

    /**
     * Clear list
     */
    clear() {
        this.head = this.tail = null;
        this._len = 0;
    }

}

/**
 * LRU Cache
 */
export default class LRU<T> {

    private _list = new LinkedList<T>()

    private _maxSize = 10

    private _lastRemovedEntry: Entry<T>

    private _map: Dictionary<Entry<T>> = {}

    constructor(maxSize: number) {
        this._maxSize = maxSize;
    }

    /**
     * @return Removed value
     */
    put(key: string | number, value: T): T {
        const list = this._list;
        const map = this._map;
        let removed = null;
        if (map[key] == null) {
            const len = list.len();
            // Reuse last removed entry
            let entry = this._lastRemovedEntry;

            if (len >= this._maxSize && len > 0) {
                // Remove the least recently used
                const leastUsedEntry = list.head;
                list.remove(leastUsedEntry);
                delete map[leastUsedEntry.key];

                removed = leastUsedEntry.value;
                this._lastRemovedEntry = leastUsedEntry;
            }

            if (entry) {
                entry.value = value;
            }
            else {
                entry = new Entry(value);
            }
            entry.key = key;
            list.insertEntry(entry);
            map[key] = entry;
        }

        return removed;
    }

    get(key: string | number): T {
        const entry = this._map[key];
        const list = this._list;
        if (entry != null) {
            // Put the latest used entry in the tail
            if (entry !== list.tail) {
                list.remove(entry);
                list.insertEntry(entry);
            }

            return entry.value;
        }
    }

    /**
     * Clear the cache
     */
    clear() {
        this._list.clear();
        this._map = {};
    }

    len() {
        return this._list.len();
    }
}