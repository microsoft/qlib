import LegendModel, { LegendOption } from './LegendModel.js';
import { ZRColor, LabelOption } from '../../util/types.js';
import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
export interface ScrollableLegendOption extends LegendOption {
    scrollDataIndex?: number;
    /**
     * Gap between each page button
     */
    pageButtonItemGap?: number;
    /**
     * Gap between page buttons group and legend items.
     */
    pageButtonGap?: number;
    pageButtonPosition?: 'start' | 'end';
    pageFormatter?: string | ((param: {
        current: number;
        total: number;
    }) => string);
    pageIcons?: {
        horizontal?: string[];
        vertical?: string[];
    };
    pageIconColor?: ZRColor;
    pageIconInactiveColor?: ZRColor;
    pageIconSize?: number;
    pageTextStyle?: LabelOption;
    animationDurationUpdate?: number;
}
declare class ScrollableLegendModel extends LegendModel<ScrollableLegendOption> {
    static type: "legend.scroll";
    type: "legend.scroll";
    /**
     * @param {number} scrollDataIndex
     */
    setScrollDataIndex(scrollDataIndex: number): void;
    init(option: ScrollableLegendOption, parentModel: Model, ecModel: GlobalModel): void;
    /**
     * @override
     */
    mergeOption(option: ScrollableLegendOption, ecModel: GlobalModel): void;
    static defaultOption: ScrollableLegendOption;
}
export default ScrollableLegendModel;
