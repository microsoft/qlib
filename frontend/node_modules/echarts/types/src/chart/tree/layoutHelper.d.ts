/**
 * @file The layout algorithm of node-link tree diagrams. Here we using Reingold-Tilford algorithm to drawing
 *       the tree.
 */
import * as layout from '../../util/layout.js';
import { TreeNode } from '../../data/Tree.js';
import TreeSeriesModel from './TreeSeries.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
interface HierNode {
    defaultAncestor: TreeLayoutNode;
    ancestor: TreeLayoutNode;
    prelim: number;
    modifier: number;
    change: number;
    shift: number;
    i: number;
    thread: TreeLayoutNode;
}
export interface TreeLayoutNode extends TreeNode {
    parentNode: TreeLayoutNode;
    hierNode: HierNode;
    children: TreeLayoutNode[];
}
/**
 * Initialize all computational message for following algorithm.
 */
export declare function init(inRoot: TreeNode): void;
/**
 * The implementation of this function was originally copied from "d3.js"
 * <https://github.com/d3/d3-hierarchy/blob/4c1f038f2725d6eae2e49b61d01456400694bac4/src/tree.js>
 * with some modifications made for this program.
 * See the license statement at the head of this file.
 *
 * Computes a preliminary x coordinate for node. Before that, this function is
 * applied recursively to the children of node, as well as the function
 * apportion(). After spacing out the children by calling executeShifts(), the
 * node is placed to the midpoint of its outermost children.
 */
export declare function firstWalk(node: TreeLayoutNode, separation: SeparationFunc): void;
/**
 * The implementation of this function was originally copied from "d3.js"
 * <https://github.com/d3/d3-hierarchy/blob/4c1f038f2725d6eae2e49b61d01456400694bac4/src/tree.js>
 * with some modifications made for this program.
 * See the license statement at the head of this file.
 *
 * Computes all real x-coordinates by summing up the modifiers recursively.
 */
export declare function secondWalk(node: TreeLayoutNode): void;
export declare function separation(cb?: SeparationFunc): SeparationFunc;
/**
 * Transform the common coordinate to radial coordinate.
 */
export declare function radialCoordinate(rad: number, r: number): {
    x: number;
    y: number;
};
/**
 * Get the layout position of the whole view.
 */
export declare function getViewRect(seriesModel: TreeSeriesModel, api: ExtensionAPI): layout.LayoutRect;
interface SeparationFunc {
    (node1: TreeLayoutNode, node2: TreeLayoutNode): number;
}
export {};
