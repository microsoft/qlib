import * as graphic from '../../util/graphic.js';
import MarkerView from './MarkerView.js';
import { HashMap } from 'zrender/lib/core/util.js';
import MarkAreaModel from './MarkAreaModel.js';
import SeriesModel from '../../model/Series.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
interface MarkAreaDrawGroup {
    group: graphic.Group;
}
export declare const dimPermutations: readonly [readonly ["x0", "y0"], readonly ["x1", "y0"], readonly ["x1", "y1"], readonly ["x0", "y1"]];
declare class MarkAreaView extends MarkerView {
    static type: string;
    type: string;
    markerGroupMap: HashMap<MarkAreaDrawGroup>;
    updateTransform(markAreaModel: MarkAreaModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    renderSeries(seriesModel: SeriesModel, maModel: MarkAreaModel, ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default MarkAreaView;
