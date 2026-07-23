import { RectLike } from 'zrender/lib/core/BoundingRect.js';
import TreemapSeriesModel from './TreemapSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { TreeNode } from '../../data/Tree.js';
import { TreemapRenderPayload, TreemapMovePayload, TreemapZoomToNodePayload } from './treemapAction.js';
export interface TreemapLayoutNode extends TreeNode {
    parentNode: TreemapLayoutNode;
    children: TreemapLayoutNode[];
    viewChildren: TreemapLayoutNode[];
}
export interface TreemapItemLayout extends RectLike {
    area: number;
    isLeafRoot: boolean;
    dataExtent: [number, number];
    borderWidth: number;
    upperHeight: number;
    upperLabelHeight: number;
    isInView: boolean;
    invisible: boolean;
    isAboveViewRoot: boolean;
}
declare const _default: {
    seriesType: string;
    reset: (seriesModel: TreemapSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload?: TreemapZoomToNodePayload | TreemapRenderPayload | TreemapMovePayload) => void;
};
/**
 * @public
 */
export default _default;
