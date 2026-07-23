import { Payload } from '../../util/types.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface TreeExpandAndCollapsePayload extends Payload {
    dataIndex: number;
}
export declare function installTreeAction(registers: EChartsExtensionInstallRegisters): void;
