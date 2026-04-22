import SeriesModel from '../../model/Series.js';
import { SeriesOption, CallbackDataParams, CircleLayoutOptionMixin, LabelLineOption, ItemStyleOption, BoxLayoutOptionMixin, OptionDataValueNumeric, SeriesEncodeOptionMixin, OptionDataItemObject, StatesOptionMixin, SeriesLabelOption, DefaultEmphasisFocus } from '../../util/types.js';
import type SeriesData from '../../data/SeriesData.js';
interface PieItemStyleOption<TCbParams = never> extends ItemStyleOption<TCbParams> {
    borderRadius?: (number | string)[] | number | string;
}
export interface PieCallbackDataParams extends CallbackDataParams {
    percent: number;
}
export interface PieStateOption<TCbParams = never> {
    itemStyle?: PieItemStyleOption<TCbParams>;
    label?: PieLabelOption;
    labelLine?: PieLabelLineOption;
}
interface PieLabelOption extends Omit<SeriesLabelOption, 'rotate' | 'position'> {
    rotate?: number | boolean | 'radial' | 'tangential';
    alignTo?: 'none' | 'labelLine' | 'edge';
    edgeDistance?: string | number;
    /**
     * @deprecated Use `edgeDistance` instead
     */
    margin?: string | number;
    bleedMargin?: number;
    distanceToLabelLine?: number;
    position?: SeriesLabelOption['position'] | 'outer' | 'inner' | 'center' | 'outside';
}
interface PieLabelLineOption extends LabelLineOption {
    /**
     * Max angle between labelLine and surface normal.
     * 0 - 180
     */
    maxSurfaceAngle?: number;
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean;
        scaleSize?: number;
    };
}
export interface PieDataItemOption extends OptionDataItemObject<OptionDataValueNumeric>, PieStateOption, StatesOptionMixin<PieStateOption, ExtraStateOption> {
    cursor?: string;
}
export interface PieSeriesOption extends Omit<SeriesOption<PieStateOption<PieCallbackDataParams>, ExtraStateOption>, 'labelLine'>, PieStateOption<PieCallbackDataParams>, Omit<CircleLayoutOptionMixin, 'center'>, BoxLayoutOptionMixin, SeriesEncodeOptionMixin {
    type?: 'pie';
    roseType?: 'radius' | 'area';
    center?: string | number | (string | number)[];
    clockwise?: boolean;
    startAngle?: number;
    endAngle?: number | 'auto';
    padAngle?: number;
    minAngle?: number;
    minShowLabelAngle?: number;
    selectedOffset?: number;
    avoidLabelOverlap?: boolean;
    percentPrecision?: number;
    stillShowZeroSum?: boolean;
    animationType?: 'expansion' | 'scale';
    animationTypeUpdate?: 'transition' | 'expansion';
    showEmptyCircle?: boolean;
    emptyCircleStyle?: PieItemStyleOption;
    data?: (OptionDataValueNumeric | OptionDataValueNumeric[] | PieDataItemOption)[];
}
declare class PieSeriesModel extends SeriesModel<PieSeriesOption> {
    static type: "series.pie";
    /**
     * @overwrite
     */
    init(option: PieSeriesOption): void;
    /**
     * @overwrite
     */
    mergeOption(): void;
    /**
     * @overwrite
     */
    getInitialData(this: PieSeriesModel): SeriesData;
    /**
     * @overwrite
     */
    getDataParams(dataIndex: number): PieCallbackDataParams;
    private _defaultLabelLine;
    static defaultOption: Omit<PieSeriesOption, 'type'>;
}
export default PieSeriesModel;
