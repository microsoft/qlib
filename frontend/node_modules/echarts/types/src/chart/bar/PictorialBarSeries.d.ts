import BaseBarSeriesModel, { BaseBarSeriesOption } from './BaseBarSeries.js';
import { OptionDataValue, ItemStyleOption, SeriesLabelOption, AnimationOptionMixin, SeriesStackOptionMixin, StatesOptionMixin, OptionDataItemObject, DefaultEmphasisFocus, SeriesEncodeOptionMixin } from '../../util/types.js';
import type Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
export interface PictorialBarStateOption {
    itemStyle?: ItemStyleOption;
    label?: SeriesLabelOption;
}
interface PictorialBarSeriesSymbolOption {
    /**
     * Customized bar shape
     */
    symbol?: string;
    /**
     * Can be ['100%', '100%'], null means auto.
     * The percent will be relative to category width. If no repeat.
     * Will be relative to symbolBoundingData.
     */
    symbolSize?: (number | string)[] | number | string;
    symbolRotate?: number;
    /**
     * Default to be auto
     */
    symbolPosition?: 'start' | 'end' | 'center';
    /**
     * Can be percent offset relative to the symbolSize
     */
    symbolOffset?: (number | string)[] | number | string;
    /**
     * start margin and end margin. Can be a number or a percent string relative to symbolSize.
     * Auto margin by default.
     */
    symbolMargin?: (number | string)[] | number | string;
    /**
     * true: means auto calculate repeat times and cut by data.
     * a number: specifies repeat times, and do not cut by data.
     * 'fixed': means auto calculate repeat times but do not cut by data.
     *
     * Otherwise means no repeat
     */
    symbolRepeat?: boolean | number | 'fixed';
    /**
     * From start to end or end to start.
     */
    symbolRepeatDirection?: 'start' | 'end';
    symbolClip?: boolean;
    /**
     * It will define the size of graphic elements.
     */
    symbolBoundingData?: number | number[];
    symbolPatternSize?: number;
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean;
    };
}
export interface PictorialBarDataItemOption extends PictorialBarSeriesSymbolOption, AnimationOptionMixin, PictorialBarStateOption, StatesOptionMixin<PictorialBarStateOption, ExtraStateOption>, OptionDataItemObject<OptionDataValue> {
    z?: number;
    cursor?: string;
}
export interface PictorialBarSeriesOption extends BaseBarSeriesOption<PictorialBarStateOption, ExtraStateOption>, PictorialBarStateOption, PictorialBarSeriesSymbolOption, SeriesStackOptionMixin, SeriesEncodeOptionMixin {
    type?: 'pictorialBar';
    coordinateSystem?: 'cartesian2d';
    data?: (PictorialBarDataItemOption | OptionDataValue | OptionDataValue[])[];
    clip?: boolean;
}
declare class PictorialBarSeriesModel extends BaseBarSeriesModel<PictorialBarSeriesOption> {
    static type: string;
    type: string;
    static dependencies: string[];
    coordinateSystem: Cartesian2D;
    hasSymbolVisual: boolean;
    defaultSymbol: string;
    static defaultOption: PictorialBarSeriesOption;
    getInitialData(option: PictorialBarSeriesOption): import("../../data/SeriesData").default<import("../../model/Model").default<any>, import("../../data/SeriesData").DefaultDataVisual>;
}
export default PictorialBarSeriesModel;
