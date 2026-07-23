import * as graphic from '../../util/graphic.js';
import { AxisPointer } from './AxisPointer.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import Element from 'zrender/lib/Element.js';
import { VerticalAlign, HorizontalAlign, CommonAxisPointerOption } from '../../util/types.js';
import { PathProps } from 'zrender/lib/graphic/Path.js';
import Model from '../../model/Model.js';
import { TextProps } from 'zrender/lib/graphic/Text.js';
interface Transform {
    x: number;
    y: number;
    rotation: number;
}
declare type AxisValue = CommonAxisPointerOption['value'];
declare type AxisPointerModel = Model<CommonAxisPointerOption>;
interface BaseAxisPointer {
    /**
     * Should be implemenented by sub-class if support `handle`.
     */
    getHandleTransform(value: AxisValue, axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel): Transform;
    /**
     * * Should be implemenented by sub-class if support `handle`.
     */
    updateHandleTransform(transform: Transform, delta: number[], axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel): Transform & {
        cursorPoint: number[];
        tooltipOption?: {
            verticalAlign?: VerticalAlign;
            align?: HorizontalAlign;
        };
    };
}
export interface AxisPointerElementOptions {
    graphicKey: string;
    pointer: PathProps & {
        type: 'Line' | 'Rect' | 'Circle' | 'Sector';
    };
    label: TextProps;
}
/**
 * Base axis pointer class in 2D.
 */
declare class BaseAxisPointer implements AxisPointer {
    private _group;
    private _lastGraphicKey;
    private _handle;
    private _dragging;
    private _lastValue;
    private _lastStatus;
    private _payloadInfo;
    /**
     * If have transition animation
     */
    private _moveAnimation;
    private _axisModel;
    private _axisPointerModel;
    private _api;
    /**
     * In px, arbitrary value. Do not set too small,
     * no animation is ok for most cases.
     */
    protected animationThreshold: number;
    /**
     * @implement
     */
    render(axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel, api: ExtensionAPI, forceRender?: boolean): void;
    /**
     * @implement
     */
    remove(api: ExtensionAPI): void;
    /**
     * @implement
     */
    dispose(api: ExtensionAPI): void;
    /**
     * @protected
     */
    determineAnimation(axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel): boolean;
    /**
     * add {pointer, label, graphicKey} to elOption
     * @protected
     */
    makeElOption(elOption: AxisPointerElementOptions, value: AxisValue, axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel, api: ExtensionAPI): void;
    /**
     * @protected
     */
    createPointerEl(group: graphic.Group, elOption: AxisPointerElementOptions, axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel): void;
    /**
     * @protected
     */
    createLabelEl(group: graphic.Group, elOption: AxisPointerElementOptions, axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel): void;
    /**
     * @protected
     */
    updatePointerEl(group: graphic.Group, elOption: AxisPointerElementOptions, updateProps: (el: Element, props: PathProps) => void): void;
    /**
     * @protected
     */
    updateLabelEl(group: graphic.Group, elOption: AxisPointerElementOptions, updateProps: (el: Element, props: PathProps) => void, axisPointerModel: AxisPointerModel): void;
    /**
     * @private
     */
    _renderHandle(value: AxisValue): void;
    private _moveHandleToValue;
    private _onHandleDragMove;
    /**
     * Throttled method.
     */
    _doDispatchAxisPointer(): void;
    private _onHandleDragEnd;
    /**
     * @private
     */
    clear(api: ExtensionAPI): void;
    /**
     * @protected
     */
    doClear(): void;
    buildLabel(xy: number[], wh: number[], xDimIndex: 0 | 1): {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}
export default BaseAxisPointer;
