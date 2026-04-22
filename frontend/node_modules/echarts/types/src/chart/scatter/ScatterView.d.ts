import SymbolDraw from '../helper/SymbolDraw.js';
import LargeSymbolDraw from '../helper/LargeSymbolDraw.js';
import ChartView from '../../view/Chart.js';
import ScatterSeriesModel from './ScatterSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import SeriesData from '../../data/SeriesData.js';
import { TaskProgressParams } from '../../core/task.js';
import Element from 'zrender/lib/Element.js';
declare class ScatterView extends ChartView {
    static readonly type = "scatter";
    type: string;
    _finished: boolean;
    _isLargeDraw: boolean;
    _symbolDraw: SymbolDraw | LargeSymbolDraw;
    render(seriesModel: ScatterSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalPrepareRender(seriesModel: ScatterSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(taskParams: TaskProgressParams, seriesModel: ScatterSeriesModel, ecModel: GlobalModel): void;
    updateTransform(seriesModel: ScatterSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void | {
        update: true;
    };
    eachRendered(cb: (el: Element) => boolean | void): void;
    _getClipShape(seriesModel: ScatterSeriesModel): import("../../coord/CoordinateSystem").CoordinateSystemClipArea;
    _updateSymbolDraw(data: SeriesData, seriesModel: ScatterSeriesModel): SymbolDraw | LargeSymbolDraw;
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(): void;
}
export default ScatterView;
