import BaseBarSeriesModel, { BaseBarSeriesOption } from './BaseBarSeries.js';
import { ItemStyleOption, OptionDataValue, SeriesStackOptionMixin, StatesOptionMixin, OptionDataItemObject, SeriesSamplingOptionMixin, SeriesLabelOption, SeriesEncodeOptionMixin, DefaultStatesMixinEmphasis, CallbackDataParams } from '../../util/types.js';
import type Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import type Polar from '../../coord/polar/Polar.js';
import SeriesData from '../../data/SeriesData.js';
import { BrushCommonSelectorsForSeries } from '../../component/brush/selector.js';
export declare type PolarBarLabelPosition = SeriesLabelOption['position'] | 'start' | 'insideStart' | 'middle' | 'end' | 'insideEnd';
export declare type BarSeriesLabelOption = Omit<SeriesLabelOption, 'position'> & {
    position?: PolarBarLabelPosition | 'outside';
};
export interface BarStateOption<TCbParams = never> {
    itemStyle?: BarItemStyleOption<TCbParams>;
    label?: BarSeriesLabelOption;
}
interface BarStatesMixin {
    emphasis?: DefaultStatesMixinEmphasis;
}
export interface BarItemStyleOption<TCbParams = never> extends ItemStyleOption<TCbParams> {
    borderRadius?: (number | string)[] | number | string;
}
export interface BarDataItemOption extends BarStateOption, StatesOptionMixin<BarStateOption, BarStatesMixin>, OptionDataItemObject<OptionDataValue> {
    cursor?: string;
}
export interface BarSeriesOption extends BaseBarSeriesOption<BarStateOption<CallbackDataParams>, BarStatesMixin>, BarStateOption<CallbackDataParams>, SeriesStackOptionMixin, SeriesSamplingOptionMixin, SeriesEncodeOptionMixin {
    type?: 'bar';
    coordinateSystem?: 'cartesian2d' | 'polar';
    clip?: boolean;
    /**
     * If use caps on two sides of bars
     * Only available on tangential polar bar
     */
    roundCap?: boolean;
    showBackground?: boolean;
    startValue?: number;
    backgroundStyle?: ItemStyleOption & {
        borderRadius?: number | number[];
    };
    data?: (BarDataItemOption | OptionDataValue | OptionDataValue[])[];
    realtimeSort?: boolean;
}
declare class BarSeriesModel extends BaseBarSeriesModel<BarSeriesOption> {
    static type: string;
    type: string;
    static dependencies: string[];
    coordinateSystem: Cartesian2D | Polar;
    getInitialData(): SeriesData;
    /**
     * @override
     */
    getProgressive(): number | false;
    /**
     * @override
     */
    getProgressiveThreshold(): number;
    brushSelector(dataIndex: number, data: SeriesData, selectors: BrushCommonSelectorsForSeries): boolean;
    static defaultOption: BarSeriesOption;
}
export default BarSeriesModel;
