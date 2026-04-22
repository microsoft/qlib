import { Group } from '../../util/graphic.js';
import ComponentView from '../../view/Component.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import VisualMapModel from './VisualMapModel.js';
declare type VisualState = VisualMapModel['stateList'][number];
declare class VisualMapView extends ComponentView {
    static type: string;
    type: string;
    autoPositionValues: {
        readonly left: 1;
        readonly right: 1;
        readonly top: 1;
        readonly bottom: 1;
    };
    ecModel: GlobalModel;
    api: ExtensionAPI;
    visualMapModel: VisualMapModel;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * @protected
     */
    render(visualMapModel: VisualMapModel, ecModel: GlobalModel, api: ExtensionAPI, payload: unknown): void;
    /**
     * @protected
     */
    renderBackground(group: Group): void;
    /**
     * @protected
     * @param targetValue can be Infinity or -Infinity
     * @param visualCluster Only can be 'color' 'opacity' 'symbol' 'symbolSize'
     * @param opts
     * @param opts.forceState Specify state, instead of using getValueState method.
     * @param opts.convertOpacityToAlpha For color gradient in controller widget.
     * @return {*} Visual value.
     */
    protected getControllerVisual(targetValue: number, visualCluster: 'color' | 'opacity' | 'symbol' | 'symbolSize', opts?: {
        forceState?: VisualState;
        convertOpacityToAlpha?: boolean;
    }): string | number;
    protected positionGroup(group: Group): void;
    protected doRender(visualMapModel: VisualMapModel, ecModel: GlobalModel, api: ExtensionAPI, payload: unknown): void;
}
export default VisualMapView;
