import ChartView from '../../view/Chart.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GraphSeriesModel from './GraphSeries.js';
declare class GraphView extends ChartView {
    static readonly type = "graph";
    readonly type = "graph";
    private _symbolDraw;
    private _lineDraw;
    private _controller;
    private _controllerHost;
    private _firstRender;
    private _model;
    private _layoutTimeout;
    private _layouting;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(seriesModel: GraphSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(): void;
    _startForceLayoutIteration(forceLayout: GraphSeriesModel['forceLayout'], layoutAnimation?: boolean): void;
    _updateController(seriesModel: GraphSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    _updateNodeAndLinkScale(): void;
    updateLayout(seriesModel: GraphSeriesModel): void;
    remove(): void;
}
export default GraphView;
