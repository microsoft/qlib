import * as graphic from '../../util/graphic.js';
import TreemapSeriesModel, { TreemapSeriesOption } from './TreemapSeries.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { TreeNode } from '../../data/Tree.js';
import { ZRElementEvent, BoxLayoutOptionMixin } from '../../util/types.js';
import Model from '../../model/Model.js';
interface OnSelectCallback {
    (node: TreeNode, e: ZRElementEvent): void;
}
interface LayoutParam {
    pos: BoxLayoutOptionMixin;
    box: {
        width: number;
        height: number;
    };
    emptyItemWidth: number;
    totalWidth: number;
    renderList: {
        node: TreeNode;
        text: string;
        width: number;
    }[];
}
declare type BreadcrumbItemStyleModel = Model<TreemapSeriesOption['breadcrumb']['itemStyle']>;
declare type BreadcrumbEmphasisItemStyleModel = Model<TreemapSeriesOption['breadcrumb']['emphasis']>;
declare type BreadcrumbTextStyleModel = Model<TreemapSeriesOption['breadcrumb']['itemStyle']['textStyle']>;
declare class Breadcrumb {
    group: graphic.Group;
    constructor(containerGroup: graphic.Group);
    render(seriesModel: TreemapSeriesModel, api: ExtensionAPI, targetNode: TreeNode, onSelect: OnSelectCallback): void;
    /**
     * Prepare render list and total width
     * @private
     */
    _prepare(targetNode: TreeNode, layoutParam: LayoutParam, textStyleModel: BreadcrumbTextStyleModel): void;
    /**
     * @private
     */
    _renderContent(seriesModel: TreemapSeriesModel, layoutParam: LayoutParam, normalStyleModel: BreadcrumbItemStyleModel, emphasisModel: BreadcrumbEmphasisItemStyleModel, textStyleModel: BreadcrumbTextStyleModel, emphasisTextStyleModel: BreadcrumbTextStyleModel, onSelect: OnSelectCallback): void;
    remove(): void;
}
export default Breadcrumb;
