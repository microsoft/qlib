import SeriesModel from '../../model/Series.js';
import { TreeNode } from '../../data/Tree.js';
import { SeriesOption, CircleLayoutOptionMixin, SeriesLabelOption, ItemStyleOption, OptionDataValue, CallbackDataParams, StatesOptionMixin, OptionDataItemObject, DefaultEmphasisFocus, SunburstColorByMixin } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
import Model from '../../model/Model.js';
interface SunburstItemStyleOption<TCbParams = never> extends ItemStyleOption<TCbParams> {
    borderRadius?: (number | string)[] | number | string;
}
interface SunburstLabelOption extends Omit<SeriesLabelOption<SunburstDataParams>, 'rotate' | 'position'> {
    rotate?: 'radial' | 'tangential' | number;
    minAngle?: number;
    silent?: boolean;
    position?: SeriesLabelOption['position'] | 'outside';
}
interface SunburstDataParams extends CallbackDataParams {
    treePathInfo: {
        name: string;
        dataIndex: number;
        value: SunburstSeriesNodeItemOption['value'];
    }[];
}
interface SunburstStatesMixin {
    emphasis?: {
        focus?: DefaultEmphasisFocus | 'descendant' | 'ancestor' | 'relative';
    };
}
export interface SunburstStateOption<TCbParams = never> {
    itemStyle?: SunburstItemStyleOption<TCbParams>;
    label?: SunburstLabelOption;
}
export interface SunburstSeriesNodeItemOption extends SunburstStateOption<SunburstDataParams>, StatesOptionMixin<SunburstStateOption<SunburstDataParams>, SunburstStatesMixin>, OptionDataItemObject<OptionDataValue> {
    nodeClick?: 'rootToNode' | 'link' | false;
    link?: string;
    target?: string;
    children?: SunburstSeriesNodeItemOption[];
    collapsed?: boolean;
    cursor?: string;
}
export interface SunburstSeriesLevelOption extends SunburstStateOption<SunburstDataParams>, StatesOptionMixin<SunburstStateOption<SunburstDataParams>, SunburstStatesMixin> {
    radius?: (number | string)[];
    /**
     * @deprecated use radius instead
     */
    r?: number | string;
    /**
     * @deprecated use radius instead
     */
    r0?: number | string;
    highlight?: {
        itemStyle?: SunburstItemStyleOption;
        label?: SunburstLabelOption;
    };
}
interface SortParam {
    dataIndex: number;
    depth: number;
    height: number;
    getValue(): number;
}
export interface SunburstSeriesOption extends SeriesOption<SunburstStateOption<SunburstDataParams>, SunburstStatesMixin>, SunburstStateOption<SunburstDataParams>, SunburstColorByMixin, CircleLayoutOptionMixin {
    type?: 'sunburst';
    clockwise?: boolean;
    startAngle?: number;
    minAngle?: number;
    /**
     * If still show when all data zero.
     */
    stillShowZeroSum?: boolean;
    /**
     * Policy of highlighting pieces when hover on one
     * Valid values: 'none' (for not downplay others), 'descendant',
     * 'ancestor', 'self'
     */
    nodeClick?: 'rootToNode' | 'link' | false;
    renderLabelForZeroData?: boolean;
    data?: SunburstSeriesNodeItemOption[];
    levels?: SunburstSeriesLevelOption[];
    animationType?: 'expansion' | 'scale';
    sort?: 'desc' | 'asc' | ((a: SortParam, b: SortParam) => number);
}
interface SunburstSeriesModel {
    getFormattedLabel(dataIndex: number, state?: 'emphasis' | 'normal' | 'highlight' | 'blur' | 'select'): string;
}
declare class SunburstSeriesModel extends SeriesModel<SunburstSeriesOption> {
    static readonly type = "series.sunburst";
    readonly type = "series.sunburst";
    ignoreStyleOnData: boolean;
    private _viewRoot;
    private _levelModels;
    getInitialData(option: SunburstSeriesOption, ecModel: GlobalModel): SeriesData<Model<any>, import("../../data/SeriesData").DefaultDataVisual>;
    optionUpdated(): void;
    getDataParams(dataIndex: number): SunburstDataParams;
    getLevelModel(node: TreeNode): Model<SunburstSeriesLevelOption>;
    static defaultOption: SunburstSeriesOption;
    getViewRoot(): TreeNode;
    resetViewRoot(viewRoot?: TreeNode): void;
    enableAriaDecal(): void;
}
export default SunburstSeriesModel;
