import { TreeNode } from '../../data/Tree.js';
/**
 * Traverse the tree from bottom to top and do something
 */
declare function eachAfter<T>(root: TreeNode, callback: (node: TreeNode, separation: T) => void, separation: T): void;
/**
 * Traverse the tree from top to bottom and do something
 */
declare function eachBefore(root: TreeNode, callback: (node: TreeNode) => void): void;
export { eachAfter, eachBefore };
