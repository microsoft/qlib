import SeriesModel from '../../model/Series.js';
import { TreeNode } from '../../data/Tree.js';
import Model from '../../model/Model.js';
import { SeriesOption, BoxLayoutOptionMixin, ItemStyleOption, LabelOption, RoamOptionMixin, CallbackDataParams, ColorString, StatesOptionMixin, OptionId, OptionName, DecalObject, SeriesLabelOption, DefaultEmphasisFocus, BlurScope } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import { LayoutRect } from '../../util/layout.js';
import SeriesData from '../../data/SeriesData.js';
declare type TreemapSeriesDataValue = number | number[];
interface BreadcrumbItemStyleOption extends ItemStyleOption {
    textStyle?: LabelOption;
}
interface TreemapSeriesLabelOption extends SeriesLabelOption {
    formatter?: string | ((params: CallbackDataParams) => string);
}
interface TreemapSeriesItemStyleOption<TCbParams = never> extends ItemStyleOption<TCbParams> {
    borderRadius?: number | number[];
    colorAlpha?: number;
    colorSaturation?: number;
    borderColorSaturation?: number;
    gapWidth?: number;
}
interface TreePathInfo {
    name: string;
    dataIndex: number;
    value: TreemapSeriesDataValue;
}
interface TreemapSeriesCallbackDataParams extends CallbackDataParams {
    /**
     * @deprecated
     */
    treePathInfo?: TreePathInfo[];
    treeAncestors?: TreePathInfo[];
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus | 'descendant' | 'ancestor';
    };
}
export interface TreemapStateOption<TCbParams = never> {
    itemStyle?: TreemapSeriesItemStyleOption<TCbParams>;
    label?: TreemapSeriesLabelOption;
    upperLabel?: TreemapSeriesLabelOption;
}
export interface TreemapSeriesVisualOption {
    /**
     * Which dimension will be applied with the visual properties.
     */
    visualDimension?: number | string;
    /**
     * @deprecated Use colorBy instead
     */
    colorMappingBy?: 'value' | 'index' | 'id';
    visualMin?: number;
    visualMax?: number;
    colorAlpha?: number[] | 'none';
    colorSaturation?: number[] | 'none';
    /**
     * A node will not be shown when its area size is smaller than this value (unit: px square).
     */
    visibleMin?: number;
    /**
     * Children will not be shown when area size of a node is smaller than this value (unit: px square).
     */
    childrenVisibleMin?: number;
}
export interface TreemapSeriesLevelOption extends TreemapSeriesVisualOption, TreemapStateOption, StatesOptionMixin<TreemapStateOption, ExtraStateOption> {
    color?: ColorString[] | 'none';
    decal?: DecalObject[] | 'none';
}
export interface TreemapSeriesNodeItemOption extends TreemapSeriesVisualOption, TreemapStateOption, StatesOptionMixin<TreemapStateOption, ExtraStateOption> {
    id?: OptionId;
    name?: OptionName;
    value?: TreemapSeriesDataValue;
    children?: TreemapSeriesNodeItemOption[];
    color?: ColorString[] | 'none';
    decal?: DecalObject[] | 'none';
    cursor?: string;
}
export interface TreemapSeriesOption extends SeriesOption<TreemapStateOption<TreemapSeriesCallbackDataParams>, ExtraStateOption>, TreemapStateOption<TreemapSeriesCallbackDataParams>, BoxLayoutOptionMixin, RoamOptionMixin, TreemapSeriesVisualOption {
    type?: 'treemap';
    /**
     * configuration in echarts2
     * @deprecated
     */
    size?: (number | string)[];
    /**
     * If sort in desc order.
     * Default to be desc. asc has strange effect
     */
    sort?: boolean | 'asc' | 'desc';
    /**
     * Size of clipped window when zooming. 'origin' or 'fullscreen'
     */
    clipWindow?: 'origin' | 'fullscreen';
    squareRatio?: number;
    /**
     * Nodes on depth from root are regarded as leaves.
     * Count from zero (zero represents only view root).
     */
    leafDepth?: number;
    drillDownIcon?: string;
    /**
     * Be effective when using zoomToNode. Specify the proportion of the
     * target node area in the view area.
     */
    zoomToNodeRatio?: number;
    /**
     * Leaf node click behaviour: 'zoomToNode', 'link', false.
     * If leafDepth is set and clicking a node which has children but
     * be on left depth, the behaviour would be changing root. Otherwise
     * use behaviour defined above.
     */
    nodeClick?: 'zoomToNode' | 'link' | false;
    breadcrumb?: BoxLayoutOptionMixin & {
        show?: boolean;
        height?: number;
        emptyItemWidth?: number;
        itemStyle?: BreadcrumbItemStyleOption;
        emphasis?: {
            disabled?: boolean;
            focus?: DefaultEmphasisFocus;
            blurScope?: BlurScope;
            itemStyle?: BreadcrumbItemStyleOption;
        };
    };
    levels?: TreemapSeriesLevelOption[];
    data?: TreemapSeriesNodeItemOption[];
}
declare class TreemapSeriesModel extends SeriesModel<TreemapSeriesOption> {
    static type: string;
    type: string;
    static layoutMode: "box";
    preventUsingHoverLayer: boolean;
    layoutInfo: LayoutRect;
    designatedVisualItemStyle: TreemapSeriesItemStyleOption;
    private _viewRoot;
    private _idIndexMap;
    private _idIndexMapCount;
    zoom: number;
    zoomLimit: {
        max?: number;
        min?: number;
    };
    static defaultOption: TreemapSeriesOption;
    /**
     * @override
     */
    getInitialData(option: TreemapSeriesOption, ecModel: GlobalModel): SeriesData<Model<any>, import("../../data/SeriesData").DefaultDataVisual>;
    optionUpdated(): void;
    /**
     * @override
     * @param {number} dataIndex
     * @param {boolean} [mutipleSeries=false]
     */
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    /**
     * Add tree path to tooltip param
     *
     * @override
     * @param {number} dataIndex
     * @return {Object}
     */
    getDataParams(dataIndex: number): TreemapSeriesCallbackDataParams;
    /**
     * @public
     * @param {Object} layoutInfo {
     *                                x: containerGroup x
     *                                y: containerGroup y
     *                                width: containerGroup width
     *                                height: containerGroup height
     *                            }
     */
    setLayoutInfo(layoutInfo: LayoutRect): void;
    /**
     * @param  {string} id
     * @return {number} index
     */
    mapIdToIndex(id: string): number;
    getViewRoot(): TreeNode;
    resetViewRoot(viewRoot?: TreeNode): void;
    enableAriaDecal(): void;
}
export default TreemapSeriesModel;
