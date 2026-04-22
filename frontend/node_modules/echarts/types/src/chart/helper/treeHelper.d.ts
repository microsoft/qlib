import SeriesModel from '../../model/Series.js';
import { TreeNode } from '../../data/Tree.js';
export declare function retrieveTargetInfo(payload: {
    type?: string;
    targetNode?: string | TreeNode;
    targetNodeId?: string;
}, validPayloadTypes: string[], seriesModel: SeriesModel): {
    node: TreeNode;
};
export declare function getPathToRoot(node: TreeNode): TreeNode[];
export declare function aboveViewRoot(viewRoot: TreeNode, node: TreeNode): boolean;
export declare function wrapTreePathInfo<T = unknown>(node: TreeNode, seriesModel: SeriesModel): {
    name: string;
    dataIndex: number;
    value: T;
}[];
