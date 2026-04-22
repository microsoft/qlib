import * as graphic from '../../util/graphic.js';
import { TreeNode } from '../../data/Tree.js';
import SunburstSeriesModel from './SunburstSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
/**
 * Sunburstce of Sunburst including Sector, Label, LabelLine
 */
declare class SunburstPiece extends graphic.Sector {
    node: TreeNode;
    private _seriesModel;
    private _ecModel;
    constructor(node: TreeNode, seriesModel: SunburstSeriesModel, ecModel: GlobalModel, api: ExtensionAPI);
    updateData(firstCreate: boolean, node: TreeNode, seriesModel: SunburstSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    _updateLabel(seriesModel: SunburstSeriesModel): void;
}
export default SunburstPiece;
