import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.emulation
import pufferlib.models
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


# A simple policy that only gets Entity, Tile, Task, and outputs Move and Attack actions.
class Baseline(pufferlib.models.Policy):
  def __init__(self, env, input_size=256, hidden_size=256):
    super().__init__(env)
    self.config = env.env.config  # nmmo config

    self.flat_observation_space = env.flat_observation_space
    self.flat_observation_structure = env.flat_observation_structure

    # obs["Tile"] has death fog and obstacle info
    proj_fc_multiplier = 4  # tile (cnn), my_agent, task, comm
    tile_attr_dim = env.structured_observation_space["Tile"].shape[1]
    self.tile_encoder = TileEncoder(input_size, tile_attr_dim)

    self.player_encoder = PlayerEncoder(input_size, hidden_size)
    task_size = env.structured_observation_space["Task"].shape[0]
    self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
    #self.comm_encoder = CommCNN(input_size)
    self.comm_encoder = CommEmbedEncoder(input_size, self.config.MAP_SIZE,
                                         self.config.COMMUNICATION_NUM_TOKENS)

    self.proj_fc = nn.Linear(proj_fc_multiplier * input_size, input_size)
    self.action_decoder = ActionDecoder(input_size, hidden_size)
    self.value_head = nn.Linear(hidden_size, 1)

  def encode_observations(self, flat_observations):
    env_outputs = pufferlib.emulation.unpack_batched_obs(flat_observations,
        self.flat_observation_space, self.flat_observation_structure)
    tile = self.tile_encoder(env_outputs["Tile"])  # 1024
    player_embeddings, my_agent = self.player_encoder(
        env_outputs["Entity"], env_outputs["AgentId"][:, 0]
    )
    task = self.task_encoder(env_outputs["Task"])
    #comm = self.comm_encoder(env_outputs["Communication"])
    comm = self.comm_encoder(env_outputs["Communication"], env_outputs["AgentId"][:, 0])
    obs = torch.cat([tile, my_agent, task, comm], dim=-1)
    obs = self.proj_fc(obs)

    return obs, (
        player_embeddings,
        env_outputs["ActionTargets"],
    )

  def decode_actions(self, hidden, lookup):
    actions = self.action_decoder(hidden, lookup)
    value = self.value_head(hidden)
    return actions, value


class TileEncoder(nn.Module):
  def __init__(self, input_size, tile_attr_dim):
    super().__init__()
    self.tile_attr_dim = tile_attr_dim
    embed_dim = 32
    self.tile_offset = torch.tensor([i * 256 for i in range(tile_attr_dim)])
    self.embedding = nn.Embedding(tile_attr_dim * 256, embed_dim)

    self.tile_conv_1 = nn.Conv2d(embed_dim * tile_attr_dim, embed_dim, kernel_size=3)
    self.tile_conv_2 = nn.Conv2d(embed_dim, 8, kernel_size=3)
    self.tile_fc = nn.Linear(8 * 11 * 11, input_size)

  def forward(self, tile):
    # row, col centering for each agent. row 112 is the agent's position
    tile[:, :, :2] -= tile[:, 112:113, :2].clone()
    # since the embedding clips the value to 0-255, we need to offset the values
    tile[:, :, :2] += 7  # row & col
    if self.tile_attr_dim > 3:
        tile[:, :, 3] = torch.clamp(tile[:, :, 3], min=0)  # death fog
    tile = self.embedding(
        tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
    )

    agents, tiles, features, embed = tile.shape
    tile = (
        tile.view(agents, tiles, features * embed)
        .transpose(1, 2)
        .view(agents, features * embed, 15, 15)
    )

    tile = F.relu(self.tile_conv_1(tile))
    tile = F.relu(self.tile_conv_2(tile))
    tile = tile.contiguous().view(agents, -1)
    tile = F.relu(self.tile_fc(tile))

    return tile


# class CommEncoder(nn.Module):
#   def __init__(self, input_size, hidden_size, comm_obs_n, token_num):
#     super().__init__()
#     self.token_num = token_num
#     self.comm_fc1 = nn.Linear((token_num+4)*comm_obs_n, hidden_size)
#     self.comm_fc2 = nn.Linear(hidden_size, input_size)

#   def forward(self, comm_obs, my_id):
#     # Input shape: (batch_size, 100, 4)
#     agent_ids = comm_obs[:, :, 0].int()
#     self_mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
#     tokens = comm_obs[:, :, 3].long()
#     comm_tensor = torch.cat((
#       self_mask.unsqueeze(-1),  # 1 indicate self
#       comm_obs[:, :, 1:3],  # row, col
#       F.one_hot(tokens, num_classes=self.token_num + 1)  # include 0
#     ), dim=2).float()
#     comm_tensor = comm_tensor.view(comm_tensor.size(0), -1)
#     comm_tensor = F.relu(self.comm_fc1(comm_tensor))
#     return self.comm_fc2(comm_tensor)

class CommEmbedEncoder(nn.Module):
  def __init__(self, input_size, map_size, token_num,
               embed_dim=16, pos_down_sample=8, comm_obs_len=32):
    super().__init__()
    self.map_size = map_size
    self.pos_down_sample = pos_down_sample
    self.token_num = token_num
    self.pos_embedding = nn.Embedding((map_size//self.pos_down_sample)**2, embed_dim)
    self.token_embedding = nn.Embedding(token_num+1, embed_dim)
    self.comm_fc = nn.Linear(embed_dim*(2*comm_obs_len+1), input_size)

  def forward(self, comm_obs, my_id):
    # Input shape: (batch_size, 32, 4)
    comm_obs = comm_obs.int()
    agent_ids = comm_obs[:, :, 0]
    row_indices = comm_obs[:, :, 1] // self.pos_down_sample
    col_indices = comm_obs[:, :, 2] // self.pos_down_sample
    tokens = comm_obs[:, :, 3]

    pos_idx = row_indices * (self.map_size//self.pos_down_sample) + col_indices
    pos_embeddings = self.pos_embedding(pos_idx)
    token_embeddings = self.token_embedding(tokens)

    # Pull out rows corresponding to the agent
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
    my_pos_embeddings = pos_embeddings[torch.arange(comm_obs.shape[0]), row_indices]

    embeddings = torch.cat((pos_embeddings, token_embeddings), dim=2)
    embeddings = embeddings.view(embeddings.size(0), -1)
    embeddings = torch.cat([my_pos_embeddings, embeddings], dim=1)
    return self.comm_fc(embeddings)

class CommCNN(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.fc = nn.Linear(32 * 5 * 5, input_size)

  def forward(self, comm_map):
    comm_map = self.features(comm_map)
    comm_map = comm_map.view(comm_map.shape[0], -1)
    return F.relu(self.fc(comm_map))


class PlayerEncoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.entity_dim = 31
    self.player_offset = torch.tensor([i * 256 for i in range(self.entity_dim)])
    self.embedding = nn.Embedding(self.entity_dim * 256, 32)

    self.agent_fc = nn.Linear(self.entity_dim * 32, hidden_size)
    self.my_agent_fc = nn.Linear(self.entity_dim * 32, input_size)

  def forward(self, agents, my_id):
    # Pull out rows corresponding to the agent
    agent_ids = agents[:, :, EntityId]
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
    )

    agent_embeddings = self.embedding(
        agents.long().clip(0, 255) + self.player_offset.to(agents.device)
    )
    batch, agent, attrs, embed = agent_embeddings.shape

    # Embed each feature separately
    agent_embeddings = agent_embeddings.view(batch, agent, attrs * embed)
    my_agent_embeddings = agent_embeddings[
        torch.arange(agents.shape[0]), row_indices
    ]

    # Project to input of recurrent size
    agent_embeddings = self.agent_fc(agent_embeddings)
    my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
    my_agent_embeddings = F.relu(my_agent_embeddings)

    return agent_embeddings, my_agent_embeddings


class TaskEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, task_size):
    super().__init__()
    self.fc = nn.Linear(task_size, input_size)

  def forward(self, task):
    return self.fc(task.clone())


class ActionDecoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.layers = nn.ModuleDict(
        {
            "attack_style": nn.Linear(hidden_size, 3),
            "attack_target": nn.Linear(hidden_size, hidden_size),
            "comm": nn.Linear(hidden_size, 127),
            "move": nn.Linear(hidden_size, 5),
        }
    )

  def apply_layer(self, layer, embeddings, mask, hidden):
    hidden = layer(hidden)
    if hidden.dim() == 2 and embeddings is not None:
      hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

    if mask is not None:
      hidden = hidden.masked_fill(mask == 0, -1e9)

    return hidden

  def forward(self, hidden, lookup):
    (
        player_embeddings,
        action_targets,
    ) = lookup

    embeddings = {
        "attack_target": player_embeddings,
    }

    action_targets = {
        "attack_style": action_targets["Attack"]["Style"],
        "attack_target": action_targets["Attack"]["Target"],
        "comm": action_targets["Comm"]["Token"],
        "move": action_targets["Move"]["Direction"],
    }

    actions = []
    for key, layer in self.layers.items():
      mask = None
      mask = action_targets[key]
      embs = embeddings.get(key)
      if embs is not None and embs.shape[1] != mask.shape[1]:
        b, _, f = embs.shape
        zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
        embs = torch.cat([embs, zeros], dim=1)

      action = self.apply_layer(layer, embs, mask, hidden)
      actions.append(action)

    return actions
