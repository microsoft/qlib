import { VectorArray } from 'zrender/lib/core/vector.js';
declare type KDTreePoint = {
    array: VectorArray;
};
declare class KDTreeNode<T> {
    left: KDTreeNode<T>;
    right: KDTreeNode<T>;
    axis: number;
    data: T;
    constructor(axis: number, data: T);
}
/**
 * @constructor
 * @alias module:echarts/data/KDTree
 * @param {Array} points List of points.
 * each point needs an array property to represent the actual data
 * @param {Number} [dimension]
 *        Point dimension.
 *        Default will use the first point's length as dimension.
 */
declare class KDTree<T extends KDTreePoint> {
    dimension: number;
    root: KDTreeNode<T>;
    private _stack;
    private _nearstNList;
    constructor(points: T[], dimension?: number);
    /**
     * Recursively build the tree.
     */
    private _buildTree;
    /**
     * Find nearest point
     * @param  target Target point
     * @param  squaredDistance Squared distance function
     * @return Nearest point
     */
    nearest(target: T, squaredDistance: (a: T, b: T) => number): T;
    _addNearest(found: number, dist: number, node: KDTreeNode<T>): void;
    /**
     * Find nearest N points
     * @param  target Target point
     * @param  N
     * @param  squaredDistance Squared distance function
     * @param  output Output nearest N points
     */
    nearestN(target: T, N: number, squaredDistance: (a: T, b: T) => number, output: T[]): T[];
}
export default KDTree;
