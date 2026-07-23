import Model from '../model/Model.js';
import SeriesData from './SeriesData.js';
import { DimensionLoose, ParsedValue, OptionDataValue, OptionDataItemObject } from '../util/types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
declare type TreeTraverseOrder = 'preorder' | 'postorder';
declare type TreeTraverseCallback<Ctx> = (this: Ctx, node: TreeNode) => boolean | void;
declare type TreeTraverseOption = {
    order?: TreeTraverseOrder;
    attr?: 'children' | 'viewChildren';
};
interface TreeNodeOption extends Pick<OptionDataItemObject<OptionDataValue>, 'name' | 'value'> {
    children?: TreeNodeOption[];
}
export declare class TreeNode {
    name: string;
    depth: number;
    height: number;
    parentNode: TreeNode;
    /**
     * Reference to list item.
     * Do not persistent dataIndex outside,
     * besause it may be changed by list.
     * If dataIndex -1,
     * this node is logical deleted (filtered) in list.
     */
    dataIndex: number;
    children: TreeNode[];
    viewChildren: TreeNode[];
    isExpand: boolean;
    readonly hostTree: Tree<Model>;
    constructor(name: string, hostTree: Tree<Model>);
    /**
     * The node is removed.
     */
    isRemoved(): boolean;
    /**
     * Travel this subtree (include this node).
     * Usage:
     *    node.eachNode(function () { ... }); // preorder
     *    node.eachNode('preorder', function () { ... }); // preorder
     *    node.eachNode('postorder', function () { ... }); // postorder
     *    node.eachNode(
     *        {order: 'postorder', attr: 'viewChildren'},
     *        function () { ... }
     *    ); // postorder
     *
     * @param options If string, means order.
     * @param options.order 'preorder' or 'postorder'
     * @param options.attr 'children' or 'viewChildren'
     * @param cb If in preorder and return false,
     *                      its subtree will not be visited.
     */
    eachNode<Ctx>(options: TreeTraverseOrder, cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    eachNode<Ctx>(options: TreeTraverseOption, cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    eachNode<Ctx>(cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    /**
     * Update depth and height of this subtree.
     */
    updateDepthAndHeight(depth: number): void;
    getNodeById(id: string): TreeNode;
    contains(node: TreeNode): boolean;
    /**
     * @param includeSelf Default false.
     * @return order: [root, child, grandchild, ...]
     */
    getAncestors(includeSelf?: boolean): TreeNode[];
    getAncestorsIndices(): number[];
    getDescendantIndices(): number[];
    getValue(dimension?: DimensionLoose): ParsedValue;
    setLayout(layout: any, merge?: boolean): void;
    /**
     * @return {Object} layout
     */
    getLayout(): any;
    getModel<T = unknown>(): Model<T>;
    getLevelModel(): Model;
    /**
     * @example
     *  setItemVisual('color', color);
     *  setItemVisual({
     *      'color': color
     *  });
     */
    setVisual(key: string, value: any): void;
    setVisual(obj: Dictionary<any>): void;
    /**
     * Get item visual
     * FIXME: make return type better
     */
    getVisual(key: string): unknown;
    getRawIndex(): number;
    getId(): string;
    /**
     * index in parent's children
     */
    getChildIndex(): number;
    /**
     * if this is an ancestor of another node
     *
     * @param node another node
     * @return if is ancestor
     */
    isAncestorOf(node: TreeNode): boolean;
    /**
     * if this is an descendant of another node
     *
     * @param node another node
     * @return if is descendant
     */
    isDescendantOf(node: TreeNode): boolean;
}
declare class Tree<HostModel extends Model = Model, LevelOption = any> {
    type: 'tree';
    root: TreeNode;
    data: SeriesData;
    hostModel: HostModel;
    levelModels: Model<LevelOption>[];
    private _nodes;
    constructor(hostModel: HostModel);
    /**
     * Travel this subtree (include this node).
     * Usage:
     *    node.eachNode(function () { ... }); // preorder
     *    node.eachNode('preorder', function () { ... }); // preorder
     *    node.eachNode('postorder', function () { ... }); // postorder
     *    node.eachNode(
     *        {order: 'postorder', attr: 'viewChildren'},
     *        function () { ... }
     *    ); // postorder
     *
     * @param options If string, means order.
     * @param options.order 'preorder' or 'postorder'
     * @param options.attr 'children' or 'viewChildren'
     * @param cb
     * @param context
     */
    eachNode<Ctx>(options: TreeTraverseOrder, cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    eachNode<Ctx>(options: TreeTraverseOption, cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    eachNode<Ctx>(cb: TreeTraverseCallback<Ctx>, context?: Ctx): void;
    getNodeByDataIndex(dataIndex: number): TreeNode;
    getNodeById(name: string): TreeNode;
    /**
     * Update item available by list,
     * when list has been performed options like 'filterSelf' or 'map'.
     */
    update(): void;
    /**
     * Clear all layouts
     */
    clearLayouts(): void;
    /**
     * data node format:
     * {
     *     name: ...
     *     value: ...
     *     children: [
     *         {
     *             name: ...
     *             value: ...
     *             children: ...
     *         },
     *         ...
     *     ]
     * }
     */
    static createTree<T extends TreeNodeOption, HostModel extends Model>(dataRoot: T, hostModel: HostModel, beforeLink?: (data: SeriesData) => void): Tree<HostModel, any>;
}
export default Tree;
