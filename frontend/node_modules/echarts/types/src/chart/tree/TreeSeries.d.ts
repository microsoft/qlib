import SeriesModel from '../../model/Series.js';
import { SeriesOption, SymbolOptionMixin, BoxLayoutOptionMixin, RoamOptionMixin, LineStyleOption, ItemStyleOption, SeriesLabelOption, OptionDataValue, StatesOptionMixin, OptionDataItemObject, CallbackDataParams, DefaultEmphasisFocus } from '../../util/types.js';
import SeriesData from '../../data/SeriesData.js';
import View from '../../coord/View.js';
import { LayoutRect } from '../../util/layout.js';
interface CurveLineStyleOption extends LineStyleOption {
    curveness?: number;
}
export interface TreeSeriesStateOption<TCbParams = never> {
    itemStyle?: ItemStyleOption<TCbParams>;
    /**
     * Line style of the edge between node and it's parent.
     */
    lineStyle?: CurveLineStyleOption;
    label?: SeriesLabelOption;
}
interface TreeStatesMixin {
    emphasis?: {
        focus?: DefaultEmphasisFocus | 'ancestor' | 'descendant' | 'relative';
        scale?: boolean;
    };
}
export interface TreeSeriesNodeItemOption extends SymbolOptionMixin<CallbackDataParams>, TreeSeriesStateOption<CallbackDataParams>, StatesOptionMixin<TreeSeriesStateOption<CallbackDataParams>, TreeStatesMixin>, OptionDataItemObject<OptionDataValue> {
    children?: TreeSeriesNodeItemOption[];
    collapsed?: boolean;
    link?: string;
    target?: string;
}
/**
 * Configuration of leaves nodes.
 */
export interface TreeSeriesLeavesOption extends TreeSeriesStateOption, StatesOptionMixin<TreeSeriesStateOption, TreeStatesMixin> {
}
export interface TreeSeriesOption extends SeriesOption<TreeSeriesStateOption, TreeStatesMixin>, TreeSeriesStateOption, SymbolOptionMixin<CallbackDataParams>, BoxLayoutOptionMixin, RoamOptionMixin {
    type?: 'tree';
    layout?: 'orthogonal' | 'radial';
    edgeShape?: 'polyline' | 'curve';
    /**
     * Available when edgeShape is polyline
     */
    edgeForkPosition?: string | number;
    nodeScaleRatio?: number;
    /**
     * The orient of orthoginal layout, can be setted to 'LR', 'TB', 'RL', 'BT'.
     * and the backward compatibility configuration 'horizontal = LR', 'vertical = TB'.
     */
    orient?: 'LR' | 'TB' | 'RL' | 'BT' | 'horizontal' | 'vertical';
    expandAndCollapse?: boolean;
    /**
     * The initial expanded depth of tree
     */
    initialTreeDepth?: number;
    leaves?: TreeSeriesLeavesOption;
    data?: TreeSeriesNodeItemOption[];
}
export interface TreeAncestors {
    name: string;
    dataIndex: number;
    value: number;
}
export interface TreeSeriesCallbackDataParams extends CallbackDataParams {
    collapsed: boolean;
    treeAncestors?: TreeAncestors[];
}
declare class TreeSeriesModel extends SeriesModel<TreeSeriesOption> {
    static readonly type = "series.tree";
    static readonly layoutMode = "box";
    coordinateSystem: View;
    layoutInfo: LayoutRect;
    hasSymbolVisual: boolean;
    ignoreStyleOnData: boolean;
    /**
     * Init a tree data structure from data in option series
     */
    getInitialData(option: TreeSeriesOption): SeriesData;
    /**
     * Make the configuration 'orient' backward compatibly, with 'horizontal = LR', 'vertical = TB'.
     * @returns {string} orient
     */
    getOrient(): "LR" | "TB" | "RL" | "BT";
    setZoom(zoom: number): void;
    setCenter(center: number[]): void;
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    getDataParams(dataIndex: number): TreeSeriesCallbackDataParams;
    static defaultOption: TreeSeriesOption;
}
export default TreeSeriesModel;
