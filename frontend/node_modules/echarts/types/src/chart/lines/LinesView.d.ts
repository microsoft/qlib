import LineDraw from '../helper/LineDraw.js';
import LargeLineDraw from '../helper/LargeLineDraw.js';
import ChartView from '../../view/Chart.js';
import LinesSeriesModel from './LinesSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { StageHandlerProgressParams } from '../../util/types.js';
import SeriesData from '../../data/SeriesData.js';
import Element from 'zrender/lib/Element.js';
declare class LinesView extends ChartView {
    static readonly type = "lines";
    readonly type = "lines";
    private _lastZlevel;
    private _finished;
    private _lineDraw;
    private _hasEffet;
    private _isPolyline;
    private _isLargeDraw;
    render(seriesModel: LinesSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalPrepareRender(seriesModel: LinesSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(taskParams: StageHandlerProgressParams, seriesModel: LinesSeriesModel, ecModel: GlobalModel): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    updateTransform(seriesModel: LinesSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): {
        readonly update: true;
    };
    _updateLineDraw(data: SeriesData, seriesModel: LinesSeriesModel): LineDraw | LargeLineDraw;
    private _showEffect;
    _clearLayer(api: ExtensionAPI): void;
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default LinesView;
