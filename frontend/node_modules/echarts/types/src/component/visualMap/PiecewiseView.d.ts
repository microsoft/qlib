import VisualMapView from './VisualMapView.js';
import type PiecewiseModel from './PiecewiseModel.js';
declare class PiecewiseVisualMapView extends VisualMapView {
    static type: "visualMap.piecewise";
    type: "visualMap.piecewise";
    visualMapModel: PiecewiseModel;
    protected doRender(): void;
    private _enableHoverLink;
    private _getItemAlign;
    private _renderEndsText;
    /**
     * @private
     * @return {Object} {peiceList, endsText} The order is the same as screen pixel order.
     */
    private _getViewData;
    private _createItemSymbol;
    private _onItemClick;
}
export default PiecewiseVisualMapView;
