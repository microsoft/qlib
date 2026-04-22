import LineDraw from '../../chart/helper/LineDraw.js';
import MarkerView from './MarkerView.js';
import MarkLineModel from './MarkLineModel.js';
import SeriesModel from '../../model/Series.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GlobalModel from '../../model/Global.js';
import { HashMap } from 'zrender/lib/core/util.js';
declare class MarkLineView extends MarkerView {
    static type: string;
    type: string;
    markerGroupMap: HashMap<LineDraw>;
    updateTransform(markLineModel: MarkLineModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    renderSeries(seriesModel: SeriesModel, mlModel: MarkLineModel, ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default MarkLineView;
