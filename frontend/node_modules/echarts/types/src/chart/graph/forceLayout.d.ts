import GlobalModel from '../../model/Global.js';
export interface ForceLayoutInstance {
    step(cb: (stopped: boolean) => void): void;
    warmUp(): void;
    setFixed(idx: number): void;
    setUnfixed(idx: number): void;
}
export default function graphForceLayout(ecModel: GlobalModel): void;
