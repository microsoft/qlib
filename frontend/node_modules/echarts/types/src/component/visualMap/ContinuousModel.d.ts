import VisualMapModel, { VisualMapOption } from './VisualMapModel.js';
import { ItemStyleOption } from '../../util/types.js';
declare type VisualState = VisualMapModel['stateList'][number];
export interface ContinousVisualMapOption extends VisualMapOption {
    align?: 'auto' | 'left' | 'right' | 'top' | 'bottom';
    /**
     * This prop effect default component type determine
     * @see echarts/component/visualMap/typeDefaulter.
     */
    calculable?: boolean;
    /**
     * selected range. In default case `range` is [min, max]
     * and can auto change along with modification of min max,
     * until user specified a range.
     */
    range?: number[];
    /**
     * Whether to enable hover highlight.
     */
    hoverLink?: boolean;
    /**
     * The extent of hovered data.
     */
    hoverLinkDataSize?: number;
    /**
     * Whether trigger hoverLink when hover handle.
     * If not specified, follow the value of `realtime`.
     */
    hoverLinkOnHandle?: boolean;
    handleIcon?: string;
    handleSize?: string | number;
    handleStyle?: ItemStyleOption;
    indicatorIcon?: string;
    indicatorSize?: string | number;
    indicatorStyle?: ItemStyleOption;
    emphasis?: {
        handleStyle?: ItemStyleOption;
    };
}
declare class ContinuousModel extends VisualMapModel<ContinousVisualMapOption> {
    static type: "visualMap.continuous";
    type: "visualMap.continuous";
    /**
     * @override
     */
    optionUpdated(newOption: ContinousVisualMapOption, isInit: boolean): void;
    /**
     * @protected
     * @override
     */
    resetItemSize(): void;
    /**
     * @private
     */
    _resetRange(): void;
    /**
     * @protected
     * @override
     */
    completeVisualOption(): void;
    /**
     * @override
     */
    setSelected(selected: number[]): void;
    /**
     * @public
     */
    getSelected(): [number, number];
    /**
     * @override
     */
    getValueState(value: number): VisualState;
    findTargetDataIndices(range: number[]): {
        seriesId: string;
        dataIndex: number[];
    }[];
    /**
     * @implement
     */
    getVisualMeta(getColorVisual: (value: number, valueState: VisualState) => string): {
        stops: {
            value: number;
            color: string;
        }[];
        outerColors: string[];
    };
    static defaultOption: ContinousVisualMapOption;
}
export default ContinuousModel;
