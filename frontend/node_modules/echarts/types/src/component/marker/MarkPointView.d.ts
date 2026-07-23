import SymbolDraw from '../../chart/helper/SymbolDraw.js';
import MarkerView from './MarkerView.js';
import SeriesModel from '../../model/Series.js';
import MarkPointModel from './MarkPointModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { HashMap } from 'zrender/lib/core/util.js';
declare class MarkPointView extends MarkerView {
    static type: string;
    type: string;
    markerGroupMap: HashMap<SymbolDraw>;
    updateTransform(markPointModel: MarkPointModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    renderSeries(seriesModel: SeriesModel, mpModel: MarkPointModel, ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default MarkPointView;
