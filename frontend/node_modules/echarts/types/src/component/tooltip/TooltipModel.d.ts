import ComponentModel from '../../model/Component.js';
import { ComponentOption, LabelOption, LineStyleOption, CommonTooltipOption, TooltipRenderMode, CallbackDataParams, TooltipOrderMode } from '../../util/types.js';
import { AxisPointerOption } from '../axisPointer/AxisPointerModel.js';
export declare type TopLevelFormatterParams = CallbackDataParams | CallbackDataParams[];
export interface TooltipOption extends CommonTooltipOption<TopLevelFormatterParams>, ComponentOption {
    mainType?: 'tooltip';
    axisPointer?: AxisPointerOption & {
        axis?: 'auto' | 'x' | 'y' | 'angle' | 'radius';
        crossStyle?: LineStyleOption & {
            textStyle?: LabelOption;
        };
    };
    /**
     * If show popup content
     */
    showContent?: boolean;
    /**
     * Trigger only works on coordinate system.
     */
    trigger?: 'item' | 'axis' | 'none';
    displayMode?: 'single' | 'multipleByCoordSys';
    /**
     * 'auto': use html by default, and use non-html if `document` is not defined
     * 'html': use html for tooltip
     * 'richText': use canvas, svg, and etc. for tooltip
     */
    renderMode?: 'auto' | TooltipRenderMode;
    /**
     * @deprecated
     * use appendTo: 'body' instead
     */
    appendToBody?: boolean;
    /**
     * If append the tooltip element to another DOM element.
     * Only available when renderMode is html
     */
    appendTo?: ((chartContainer: HTMLElement) => HTMLElement | undefined | null) | string | HTMLElement;
    /**
     * Specify the class name of tooltip element
     * Only available when renderMode is html
     */
    className?: string;
    order?: TooltipOrderMode;
}
declare class TooltipModel extends ComponentModel<TooltipOption> {
    static type: "tooltip";
    type: "tooltip";
    static dependencies: string[];
    static defaultOption: TooltipOption;
}
export default TooltipModel;
