import type ExtensionAPI from '../../core/ExtensionAPI.js';
import type { TooltipOption } from './TooltipModel.js';
import Model from '../../model/Model.js';
import type { ZRColor } from '../../util/types.js';
interface TooltipContentOption {
    /**
     * Specify target container of the tooltip element.
     * Can either be an HTMLElement, CSS selector string, or a function that returns an HTMLElement.
     */
    appendTo: ((chartContainer: HTMLElement) => HTMLElement | undefined | null) | HTMLElement | string;
}
declare class TooltipHTMLContent {
    el: HTMLDivElement;
    private _api;
    private _container;
    private _show;
    private _styleCoord;
    private _enterable;
    private _zr;
    private _alwaysShowContent;
    private _hideTimeout;
    /**
     * Hide delay time
     */
    private _hideDelay;
    private _inContent;
    private _firstShow;
    private _longHide;
    /**
     * Record long-time hide
     */
    private _longHideTimeout;
    constructor(api: ExtensionAPI, opt: TooltipContentOption);
    /**
     * Update when tooltip is rendered
     */
    update(tooltipModel: Model<TooltipOption>): void;
    show(tooltipModel: Model<TooltipOption>, nearPointColor: ZRColor): void;
    setContent(content: string | HTMLElement | HTMLElement[], markers: unknown, tooltipModel: Model<TooltipOption>, borderColor?: ZRColor, arrowPosition?: TooltipOption['position']): void;
    setEnterable(enterable: boolean): void;
    getSize(): number[];
    moveTo(zrX: number, zrY: number): void;
    /**
     * when `alwaysShowContent` is true,
     * move the tooltip after chart resized
     */
    _moveIfResized(): void;
    hide(): void;
    hideLater(time?: number): void;
    isShow(): boolean;
    dispose(): void;
}
export default TooltipHTMLContent;
