import ChartView from '../../view/Chart.js';
import SunburstPiece from './SunburstPiece.js';
import SunburstSeriesModel from './SunburstSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { TreeNode } from '../../data/Tree.js';
interface DrawTreeNode extends TreeNode {
    parentNode: DrawTreeNode;
    piece: SunburstPiece;
    children: DrawTreeNode[];
}
declare class SunburstView extends ChartView {
    static readonly type = "sunburst";
    readonly type = "sunburst";
    seriesModel: SunburstSeriesModel;
    api: ExtensionAPI;
    ecModel: GlobalModel;
    virtualPiece: SunburstPiece;
    private _oldChildren;
    render(seriesModel: SunburstSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: any): void;
    /**
     * @private
     */
    _initEvents(): void;
    /**
     * @private
     */
    _rootToNode(node: DrawTreeNode): void;
    /**
     * @implement
     */
    containPoint(point: number[], seriesModel: SunburstSeriesModel): boolean;
}
export default SunburstView;
