import ChartView from '../../view/Chart.js';
import { TreeNode } from '../../data/Tree.js';
import TreemapSeriesModel from './TreemapSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { TreemapRootToNodePayload, TreemapMovePayload, TreemapRenderPayload, TreemapZoomToNodePayload } from './treemapAction.js';
interface FoundTargetInfo {
    node: TreeNode;
    offsetX?: number;
    offsetY?: number;
}
declare class TreemapView extends ChartView {
    static type: string;
    type: string;
    private _containerGroup;
    private _breadcrumb;
    private _controller;
    private _controllerHost;
    private _oldTree;
    private _state;
    private _storage;
    seriesModel: TreemapSeriesModel;
    api: ExtensionAPI;
    ecModel: GlobalModel;
    /**
     * @override
     */
    render(seriesModel: TreemapSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: TreemapZoomToNodePayload | TreemapRenderPayload | TreemapMovePayload | TreemapRootToNodePayload): void;
    private _giveContainerGroup;
    private _doRender;
    private _doAnimation;
    private _resetController;
    private _clearController;
    private _onPan;
    private _onZoom;
    private _initEvents;
    private _renderBreadcrumb;
    /**
     * @override
     */
    remove(): void;
    dispose(): void;
    private _zoomToNode;
    private _rootToNode;
    /**
     * @public
     * @param {number} x Global coord x.
     * @param {number} y Global coord y.
     * @return {Object} info If not found, return undefined;
     * @return {number} info.node Target node.
     * @return {number} info.offsetX x refer to target node.
     * @return {number} info.offsetY y refer to target node.
     */
    findTarget(x: number, y: number): FoundTargetInfo;
}
export default TreemapView;
