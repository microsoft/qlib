import ChartView from '../../view/Chart.js';
import CandlestickSeriesModel from './CandlestickSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { StageHandlerProgressParams } from '../../util/types.js';
import Element from 'zrender/lib/Element.js';
declare class CandlestickView extends ChartView {
    static readonly type = "candlestick";
    readonly type = "candlestick";
    private _isLargeDraw;
    private _data;
    private _progressiveEls;
    render(seriesModel: CandlestickSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalPrepareRender(seriesModel: CandlestickSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(params: StageHandlerProgressParams, seriesModel: CandlestickSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    _updateDrawMode(seriesModel: CandlestickSeriesModel): void;
    _renderNormal(seriesModel: CandlestickSeriesModel): void;
    _renderLarge(seriesModel: CandlestickSeriesModel): void;
    _incrementalRenderNormal(params: StageHandlerProgressParams, seriesModel: CandlestickSeriesModel): void;
    _incrementalRenderLarge(params: StageHandlerProgressParams, seriesModel: CandlestickSeriesModel): void;
    remove(ecModel: GlobalModel): void;
    _clear(): void;
}
export default CandlestickView;
