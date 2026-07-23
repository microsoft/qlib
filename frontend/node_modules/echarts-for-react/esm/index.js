import { __extends } from "tslib";
import * as echarts from 'echarts';
import EChartsReactCore from './core';
// export the Component the echarts Object.
var EChartsReact = /** @class */ (function (_super) {
    __extends(EChartsReact, _super);
    function EChartsReact(props) {
        var _this = _super.call(this, props) || this;
        // 初始化为 echarts 整个包
        _this.echarts = echarts;
        return _this;
    }
    return EChartsReact;
}(EChartsReactCore));
export default EChartsReact;
//# sourceMappingURL=index.js.map