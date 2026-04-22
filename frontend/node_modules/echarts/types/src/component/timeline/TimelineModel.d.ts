import ComponentModel from '../../model/Component.js';
import SeriesData from '../../data/SeriesData.js';
import { ComponentOption, BoxLayoutOptionMixin, LayoutOrient, SymbolOptionMixin, LineStyleOption, ItemStyleOption, LabelOption, OptionDataValue, ZRColor, ColorString, CommonTooltipOption, CallbackDataParams, ZREasing } from '../../util/types.js';
import Model from '../../model/Model.js';
import GlobalModel, { GlobalModelSetOptionOpts } from '../../model/Global.js';
export interface TimelineControlStyle extends ItemStyleOption {
    show?: boolean;
    showPlayBtn?: boolean;
    showPrevBtn?: boolean;
    showNextBtn?: boolean;
    itemSize?: number;
    itemGap?: number;
    position?: 'left' | 'right' | 'top' | 'bottom';
    playIcon?: string;
    stopIcon?: string;
    prevIcon?: string;
    nextIcon?: string;
    playBtnSize?: number | string;
    stopBtnSize?: number | string;
    nextBtnSize?: number | string;
    prevBtnSize?: number | string;
}
export interface TimelineCheckpointStyle extends ItemStyleOption, SymbolOptionMixin {
    animation?: boolean;
    animationDuration?: number;
    animationEasing?: ZREasing;
}
interface TimelineLineStyleOption extends LineStyleOption {
    show?: boolean;
}
interface TimelineLabelOption extends Omit<LabelOption, 'position'> {
    show?: boolean;
    position?: 'auto' | 'left' | 'right' | 'top' | 'bottom' | number;
    interval?: 'auto' | number;
    formatter?: string | ((value: string | number, index: number) => string);
}
export interface TimelineDataItemOption extends SymbolOptionMixin {
    value?: OptionDataValue;
    itemStyle?: ItemStyleOption;
    label?: TimelineLabelOption;
    checkpointStyle?: TimelineCheckpointStyle;
    emphasis?: {
        itemStyle?: ItemStyleOption;
        label?: TimelineLabelOption;
        checkpointStyle?: TimelineCheckpointStyle;
    };
    progress?: {
        lineStyle?: TimelineLineStyleOption;
        itemStyle?: ItemStyleOption;
        label?: TimelineLabelOption;
    };
    tooltip?: boolean;
}
export interface TimelineOption extends ComponentOption, BoxLayoutOptionMixin, SymbolOptionMixin {
    mainType?: 'timeline';
    backgroundColor?: ZRColor;
    borderColor?: ColorString;
    borderWidth?: number;
    tooltip?: CommonTooltipOption<CallbackDataParams> & {
        trigger?: 'item';
    };
    show?: boolean;
    axisType?: 'category' | 'time' | 'value';
    currentIndex?: number;
    autoPlay?: boolean;
    rewind?: boolean;
    loop?: boolean;
    playInterval?: number;
    realtime?: boolean;
    controlPosition?: 'left' | 'right' | 'top' | 'bottom';
    padding?: number | number[];
    orient?: LayoutOrient;
    inverse?: boolean;
    replaceMerge?: GlobalModelSetOptionOpts['replaceMerge'];
    lineStyle?: TimelineLineStyleOption;
    itemStyle?: ItemStyleOption;
    checkpointStyle?: TimelineCheckpointStyle;
    controlStyle?: TimelineControlStyle;
    label?: TimelineLabelOption;
    emphasis?: {
        lineStyle?: TimelineLineStyleOption;
        itemStyle?: ItemStyleOption;
        checkpointStyle?: TimelineCheckpointStyle;
        controlStyle?: TimelineControlStyle;
        label?: TimelineLabelOption;
    };
    progress?: {
        lineStyle?: TimelineLineStyleOption;
        itemStyle?: ItemStyleOption;
        label?: TimelineLabelOption;
    };
    data?: (OptionDataValue | TimelineDataItemOption)[];
}
declare class TimelineModel extends ComponentModel<TimelineOption> {
    static type: string;
    type: string;
    layoutMode: string;
    private _data;
    private _names;
    /**
     * @override
     */
    init(option: TimelineOption, parentModel: Model, ecModel: GlobalModel): void;
    /**
     * @override
     */
    mergeOption(option: TimelineOption): void;
    setCurrentIndex(currentIndex: number): void;
    /**
     * @return {number} currentIndex
     */
    getCurrentIndex(): number;
    /**
     * @return {boolean}
     */
    isIndexMax(): boolean;
    /**
     * @param {boolean} state true: play, false: stop
     */
    setPlayState(state: boolean): void;
    /**
     * @return {boolean} true: play, false: stop
     */
    getPlayState(): boolean;
    /**
     * @private
     */
    _initData(): void;
    getData(): SeriesData<TimelineModel, import("../../data/SeriesData").DefaultDataVisual>;
    /**
     * @public
     * @return {Array.<string>} categoreis
     */
    getCategories(): string[];
    /**
     * @protected
     */
    static defaultOption: TimelineOption;
}
export default TimelineModel;
