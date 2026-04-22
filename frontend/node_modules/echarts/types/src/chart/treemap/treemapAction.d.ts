import { Payload } from '../../util/types.js';
import { TreeNode } from '../../data/Tree.js';
import { RectLike } from 'zrender/lib/core/BoundingRect.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface TreemapZoomToNodePayload extends Payload {
    type: 'treemapZoomToNode';
}
export interface TreemapRenderPayload extends Payload {
    type: 'treemapRender';
    rootRect?: RectLike;
}
export interface TreemapMovePayload extends Payload {
    type: 'treemapMove';
    rootRect?: RectLike;
}
export interface TreemapRootToNodePayload extends Payload {
    type: 'treemapRootToNode';
    targetNode?: TreeNode | string;
    targetNodeId?: string;
    direction?: 'rollUp' | 'drillDown';
}
export declare function installTreemapAction(registers: EChartsExtensionInstallRegisters): void;
