"""Plasticidade intrínseca lenta das Kenyon para manter atividade esparsa."""

import numpy as np


def kenyon_homeostasis(model, observed_hz: float, target_hz: float = 1.6):
    indices = model.regions["kc"]
    current = np.asarray(model.group.I_base[:])[indices]
    change = np.clip(0.04 * (target_hz - observed_hz), -0.2, 0.2)
    model.group.I_base[indices] = np.clip(current + change, -3, 3)


class MotorPlasticity:
    """R-STDP local em blocos, com dopamina derivada de spikes de DAN.

    Os traços são por neurônio e por sinapse; os pesos mantêm o sinal de Dale.
    A resolução temporal dos pares pré/pós é o bloco da malha fechada.
    """

    def __init__(self, model, config, reward_mode="coupled"):
        if reward_mode not in ("coupled", "shuffled"):
            raise ValueError("controle de recompensa inválido")
        self.model = model
        self.config = config
        self.reward_mode = reward_mode
        self.edges = np.asarray(model.plastic_edges, dtype=np.int32)
        self.pre = np.asarray(model.synapses.i[:], dtype=np.int32)[self.edges]
        self.post = np.asarray(model.synapses.j[:], dtype=np.int32)[self.edges]
        leg_index = np.full(len(model.group), -1, dtype=np.int8)
        for index, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            leg_index[model.regions[f"vnc_{leg}"]] = index
        self.interleg = ((leg_index[self.pre] >= 0) & (leg_index[self.post] >= 0)
                         & (leg_index[self.pre] != leg_index[self.post])
                         & model.inhibitory[self.pre])
        self.sign = np.where(model.inhibitory[self.pre], -1.0, 1.0).astype(np.float32)
        self.weight = np.abs(np.asarray(model.synapses.w[:], dtype=np.float32)[self.edges])
        self.initial = self.weight.copy()
        self.eligibility = np.zeros(len(self.edges), dtype=np.float32)
        self.pre_trace = np.zeros(len(model.group), dtype=np.float32)
        self.post_trace = np.zeros(len(model.group), dtype=np.float32)
        self.last_count = np.zeros(len(model.group), dtype=np.int64)
        self.dan_fast = 0.0
        self.dan_slow = 0.0
        self.oct_fast = 0.0
        self.elapsed = 0.0
        self.next_weight_update = 0.1
        self.next_normalization = 1.0
        self.rng = np.random.default_rng(config.seed + 70000)
        self.dan_history = np.zeros(1000, dtype=np.float32)
        self.minimum_reward_lag = int(np.ceil(5000 / config.block_ms))
        self.history_cursor = 0
        self.last_dopamine = 0.0
        self.last_octopamine = 0.0
        self._ensure_structural_state()

    def _ensure_structural_state(self):
        if hasattr(self,'active'):return
        for name,value in (('active',np.ones(len(self.edges),bool)),('usage',np.zeros(len(self.edges),np.float32)),('importance',np.zeros(len(self.edges),np.float32)),('weak_time',np.zeros(len(self.edges),np.float32))):
            if not hasattr(self,name):setattr(self,name,value)

    def _prune(self):
        if not self.config.pruning_enabled:return
        candidate=np.flatnonzero(self.active&(self.sign>0)&(self.weak_time>=self.config.prune_after_s)&(self.importance<.05))
        candidate=candidate[np.argsort(self.weight[candidate]/self.initial[candidate],kind='stable')]
        exc=self.sign>0
        original=np.bincount(self.post[exc],minlength=len(self.model.group))
        remaining=np.bincount(self.post[exc&self.active],minlength=len(self.model.group))
        minimum=np.maximum(8,np.ceil(original*(1-self.config.prune_max_fraction))).astype(int)
        budget=max(1,int(.01*len(self.edges)))
        for edge in candidate:
            target=self.post[edge]
            if remaining[target]<=minimum[target]:continue
            self.active[edge]=False;self.weight[edge]=0;self.eligibility[edge]=0
            remaining[target]-=1;budget-=1
            if not budget:break

    def update(self, count, dt, enabled=True):
        self._ensure_structural_state()
        count = np.asarray(count, dtype=np.int64)
        spikes = (count - self.last_count).astype(np.float32)
        self.last_count = count.copy()
        self.elapsed += dt
        decay = np.exp(-dt / 0.020)
        self.pre_trace *= decay
        self.post_trace *= decay
        self.eligibility *= np.exp(-dt / self.config.eligibility_tau_s)
        if enabled:
            # Ordem pré/pós aproximada por blocos; LTD é ligeiramente maior
            # para conter deriva quando a dopamina oscila em torno do basal.
            self.eligibility += (self.pre_trace[self.pre] * spikes[self.post]
                                 - 1.05 * self.post_trace[self.post] * spikes[self.pre])
        self.pre_trace += spikes
        self.post_trace += spikes
        dan_hz = float(spikes[self.model.regions["dan"]].mean() / dt)
        oct_hz = float(spikes[self.model.regions["oct"]].mean() / dt)
        self.dan_fast += (1 - np.exp(-dt / 0.5)) * (dan_hz - self.dan_fast)
        self.dan_slow += (1 - np.exp(-dt / 10.0)) * (dan_hz - self.dan_slow)
        self.oct_fast += (1 - np.exp(-dt / 0.5)) * (oct_hz - self.oct_fast)
        dopamine = float(np.clip((self.dan_fast - self.dan_slow) / 20.0, -0.3, 1.0))
        self.dan_history[self.history_cursor % len(self.dan_history)] = dopamine
        self.history_cursor += 1
        if self.reward_mode == "shuffled":
            # Atraso >=5 s, maior que o traço de elegibilidade (0,5–2 s).
            # Nos primeiros 5 s não há amostra desacoplada disponível.
            if self.history_cursor <= self.minimum_reward_lag:
                dopamine = 0.0
            else:
                lag = int(self.rng.integers(self.minimum_reward_lag,
                                           min(self.history_cursor, len(self.dan_history)) + 1))
                dopamine = float(self.dan_history[(self.history_cursor - lag) % len(self.dan_history)])
        self.last_dopamine = dopamine
        self.last_octopamine = oct_hz
        if enabled and self.config.pruning_enabled:
            activity=np.clip(np.minimum(self.pre_trace[self.pre],self.post_trace[self.post]),0,1)
            self.usage+=-np.expm1(-dt/5)*(activity-self.usage)
            credit=np.abs(dopamine*self.eligibility);credit/=1+credit
            self.importance+=-np.expm1(-dt/30)*(credit-self.importance)
            weak=(self.weight<self.initial*self.config.prune_weak_ratio)&(self.usage<.01)&(self.importance<.05)&self.active
            self.weak_time=np.where(weak,self.weak_time+dt,0).astype(np.float32)
        update_due=self.elapsed+1e-9>=self.next_weight_update
        normalize_due=self.elapsed+1e-9>=self.next_normalization
        # Disabled intervals must not accumulate overdue writes on reactivation.
        if update_due:self.next_weight_update=(np.floor((self.elapsed+1e-9)/.1)+1)*.1
        if normalize_due:self.next_normalization=np.floor(self.elapsed+1e-9)+1
        if enabled and update_due:
            exploration = np.clip(self.oct_fast / 15.0, 0.2, 3.0)
            self.weight += self.config.learning_rate * exploration * dopamine * self.eligibility
            self.weight[self.interleg] += (self.config.vnc_hebb_rate
                                            * self.eligibility[self.interleg])
            np.clip(self.weight, self.initial * 0.2, self.initial * 3.0, out=self.weight)
            self.weight[~self.active]=0
            if normalize_due:
                self._prune()
                self._normalize()
            self.model.synapses.w[self.edges] = self.sign * self.weight

    def _normalize(self):
        """Conserva separadamente a entrada E e I plástica de cada neurônio."""
        for sign in (-1.0, 1.0):
            mask = (self.sign == sign)&self.active
            target = self.post[mask]
            original=self.sign==sign
            baseline = np.bincount(self.post[original], weights=self.initial[original],
                                   minlength=len(self.model.group))
            current = np.bincount(target, weights=self.weight[mask],
                                  minlength=len(self.model.group))
            scale = np.clip(baseline[target] / np.maximum(current[target], 1e-9), 0.9, 1.1)
            self.weight[mask] *= scale.astype(np.float32)
        np.clip(self.weight, self.initial * 0.2, self.initial * 3.0, out=self.weight)
        self.weight[~self.active]=0

    def summary(self):
        ratio = self.weight / self.initial
        return {"count": len(self.edges), "min": float(ratio.min()),
                "p05": float(np.quantile(ratio, .05)),
                "p50": float(np.median(ratio)), "p95": float(np.quantile(ratio, .95)),
                "max": float(ratio.max()), "dopamine": self.last_dopamine,
                "floor_fraction": float(np.mean(ratio <= .201)),
                "ceiling_fraction": float(np.mean(ratio >= 2.999)),
                "octopamine_hz": self.oct_fast,
                "pruned":int((~self.active).sum()),"pruning_enabled":self.config.pruning_enabled}
