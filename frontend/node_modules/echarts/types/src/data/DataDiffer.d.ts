import { ArrayLike } from 'zrender/lib/core/types.js';
declare type DiffKeyGetter<CTX = unknown> = (this: DataDiffer<CTX>, value: unknown, index: number) => string;
declare type DiffCallbackAdd = (newIndex: number) => void;
declare type DiffCallbackUpdate = (newIndex: number, oldIndex: number) => void;
declare type DiffCallbackRemove = (oldIndex: number) => void;
declare type DiffCallbackUpdateManyToOne = (newIndex: number, oldIndex: number[]) => void;
declare type DiffCallbackUpdateOneToMany = (newIndex: number[], oldIndex: number) => void;
declare type DiffCallbackUpdateManyToMany = (newIndex: number[], oldIndex: number[]) => void;
export declare type DataDiffMode = 'oneToOne' | 'multiple';
declare class DataDiffer<CTX = unknown> {
    private _old;
    private _new;
    private _oldKeyGetter;
    private _newKeyGetter;
    private _add;
    private _update;
    private _updateManyToOne;
    private _updateOneToMany;
    private _updateManyToMany;
    private _remove;
    private _diffModeMultiple;
    readonly context: CTX;
    /**
     * @param context Can be visited by this.context in callback.
     */
    constructor(oldArr: ArrayLike<unknown>, newArr: ArrayLike<unknown>, oldKeyGetter?: DiffKeyGetter<CTX>, newKeyGetter?: DiffKeyGetter<CTX>, context?: CTX, diffMode?: DataDiffMode);
    /**
     * Callback function when add a data
     */
    add(func: DiffCallbackAdd): this;
    /**
     * Callback function when update a data
     */
    update(func: DiffCallbackUpdate): this;
    /**
     * Callback function when update a data and only work in `cbMode: 'byKey'`.
     */
    updateManyToOne(func: DiffCallbackUpdateManyToOne): this;
    /**
     * Callback function when update a data and only work in `cbMode: 'byKey'`.
     */
    updateOneToMany(func: DiffCallbackUpdateOneToMany): this;
    /**
     * Callback function when update a data and only work in `cbMode: 'byKey'`.
     */
    updateManyToMany(func: DiffCallbackUpdateManyToMany): this;
    /**
     * Callback function when remove a data
     */
    remove(func: DiffCallbackRemove): this;
    execute(): void;
    private _executeOneToOne;
    /**
     * For example, consider the case:
     * oldData: [o0, o1, o2, o3, o4, o5, o6, o7],
     * newData: [n0, n1, n2, n3, n4, n5, n6, n7, n8],
     * Where:
     *     o0, o1, n0 has key 'a' (many to one)
     *     o5, n4, n5, n6 has key 'b' (one to many)
     *     o2, n1 has key 'c' (one to one)
     *     n2, n3 has key 'd' (add)
     *     o3, o4 has key 'e' (remove)
     *     o6, o7, n7, n8 has key 'f' (many to many, treated as add and remove)
     * Then:
     *     (The order of the following directives are not ensured.)
     *     this._updateManyToOne(n0, [o0, o1]);
     *     this._updateOneToMany([n4, n5, n6], o5);
     *     this._update(n1, o2);
     *     this._remove(o3);
     *     this._remove(o4);
     *     this._remove(o6);
     *     this._remove(o7);
     *     this._add(n2);
     *     this._add(n3);
     *     this._add(n7);
     *     this._add(n8);
     */
    private _executeMultiple;
    private _performRestAdd;
    private _initIndexMap;
}
export default DataDiffer;
