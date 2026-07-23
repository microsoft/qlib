import * as graphic from '../../util/graphic.js';
import ChartView from '../../view/Chart.js';
import HeatmapSeriesModel from './HeatmapSeries.js';
import type GlobalModel from '../../model/Global.js';
import type ExtensionAPI from '../../core/ExtensionAPI.js';
import type VisualMapModel from '../../component/visualMap/VisualMapModel.js';
import { CoordinateSystem } from '../../coord/CoordinateSystem.js';
import { StageHandlerProgressParams } from '../../util/types.js';
import type Element from 'zrender/lib/Element.js';
interface GeoLikeCoordSys extends CoordinateSystem {
    dimensions: ['lng', 'lat'];
    getViewRect(): graphic.BoundingRect;
}
declare class HeatmapView extends ChartView {
    static readonly type = "heatmap";
    readonly type = "heatmap";
    private _hmLayer;
    private _progressiveEls;
    render(seriesModel: HeatmapSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalPrepareRender(seriesModel: HeatmapSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(params: StageHandlerProgressParams, seriesModel: HeatmapSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    _renderOnCartesianAndCalendar(seriesModel: HeatmapSeriesModel, api: ExtensionAPI, start: number, end: number, incremental?: boolean): void;
    _renderOnGeo(geo: GeoLikeCoordSys, seriesModel: HeatmapSeriesModel, visualMapModel: VisualMapModel, api: ExtensionAPI): void;
}
export default HeatmapView;
