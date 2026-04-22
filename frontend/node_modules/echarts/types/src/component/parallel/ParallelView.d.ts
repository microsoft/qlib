import GlobalModel from '../../model/Global.js';
import ParallelModel from '../../coord/parallel/ParallelModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import ComponentView from '../../view/Component.js';
import { ParallelAxisExpandPayload } from '../axis/parallelAxisAction.js';
declare class ParallelView extends ComponentView {
    static type: string;
    readonly type: string;
    _model: ParallelModel;
    private _api;
    _mouseDownPoint: number[];
    private _handlers;
    render(parallelModel: ParallelModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * @internal
     * @param {Object} [opt] If null, cancel the last action triggering for debounce.
     */
    _throttledDispatchExpand(this: ParallelView, opt: Omit<ParallelAxisExpandPayload, 'type'>): void;
    /**
     * @internal
     */
    _dispatchExpand(opt: Omit<ParallelAxisExpandPayload, 'type'>): void;
}
export default ParallelView;
