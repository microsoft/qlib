import DataZoomModel, { DataZoomOption } from './DataZoomModel.js';
import { BoxLayoutOptionMixin, ZRColor, LineStyleOption, AreaStyleOption, ItemStyleOption, LabelOption } from '../../util/types.js';
interface SliderHandleLabelOption {
    show?: boolean;
}
export interface SliderDataZoomOption extends DataZoomOption, BoxLayoutOptionMixin {
    show?: boolean;
    /**
     * Slider dataZoom don't support textStyle
     */
    /**
     * Background of slider zoom component
     */
    backgroundColor?: ZRColor;
    /**
     * @deprecated Use borderColor instead
     */
    /**
     * border color of the box. For compatibility,
     * if dataBackgroundColor is set, borderColor
     * is ignored.
     */
    borderColor?: ZRColor;
    /**
     * Border radius of the box.
     */
    borderRadius?: number | number[];
    dataBackground?: {
        lineStyle?: LineStyleOption;
        areaStyle?: AreaStyleOption;
    };
    selectedDataBackground?: {
        lineStyle?: LineStyleOption;
        areaStyle?: AreaStyleOption;
    };
    /**
     * Color of selected area.
     */
    fillerColor?: ZRColor;
    /**
     * @deprecated Use handleStyle instead
     */
    handleIcon?: string;
    handleLabel?: SliderHandleLabelOption;
    /**
     * number: height of icon. width will be calculated according to the aspect of icon.
     * string: percent of the slider height. width will be calculated according to the aspect of icon.
     */
    handleSize?: string | number;
    handleStyle?: ItemStyleOption;
    /**
     * Icon to indicate it is a draggable panel.
     */
    moveHandleIcon?: string;
    moveHandleStyle?: ItemStyleOption;
    /**
     * Height of handle rect. Can be a percent string relative to the slider height.
     */
    moveHandleSize?: number;
    labelPrecision?: number | 'auto';
    labelFormatter?: string | ((value: number, valueStr: string) => string);
    showDetail?: boolean;
    showDataShadow?: 'auto' | boolean;
    zoomLock?: boolean;
    textStyle?: LabelOption;
    /**
     * If eable select by brushing
     */
    brushSelect?: boolean;
    brushStyle?: ItemStyleOption;
    emphasis?: {
        handleLabel: SliderHandleLabelOption;
        handleStyle?: ItemStyleOption;
        moveHandleStyle?: ItemStyleOption;
    };
}
declare class SliderZoomModel extends DataZoomModel<SliderDataZoomOption> {
    static readonly type = "dataZoom.slider";
    type: string;
    static readonly layoutMode = "box";
    static defaultOption: SliderDataZoomOption;
}
export default SliderZoomModel;
