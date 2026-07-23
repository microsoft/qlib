import ExtensionAPI from '../../core/ExtensionAPI.js';
import { TooltipOption } from './TooltipModel.js';
import { ZRColor } from '../../util/types.js';
import Model from '../../model/Model.js';
import ZRText from 'zrender/lib/graphic/Text.js';
import { TooltipMarkupStyleCreator } from './tooltipMarkup.js';
declare class TooltipRichContent {
    private _zr;
    private _show;
    private _styleCoord;
    private _hideTimeout;
    private _alwaysShowContent;
    private _enterable;
    private _inContent;
    private _hideDelay;
    el: ZRText;
    constructor(api: ExtensionAPI);
    /**
     * Update when tooltip is rendered
     */
    update(tooltipModel: Model<TooltipOption>): void;
    show(): void;
    /**
     * Set tooltip content
     */
    setContent(content: string | HTMLElement | HTMLElement[], markupStyleCreator: TooltipMarkupStyleCreator, tooltipModel: Model<TooltipOption>, borderColor: ZRColor, arrowPosition: TooltipOption['position']): void;
    setEnterable(enterable?: boolean): void;
    getSize(): number[];
    moveTo(x: number, y: number): void;
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
export default TooltipRichContent;
