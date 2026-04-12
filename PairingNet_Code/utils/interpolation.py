import torch


def bilinear_interpolation(x, y, feature_map, eps=1e-6):
    device = feature_map.device

    x = x.to(device)
    y = y.to(device)

    eps = torch.tensor(eps, device=device)

    torch.clamp_(x, min=eps, max=feature_map.shape[-1] - 1)
    torch.clamp_(y, min=eps, max=feature_map.shape[-1] - 1)

    bs, c, _, _ = feature_map.shape
    nums = x.shape[1] if x.dim() > 1 else len(x)

    x_1 = torch.floor(x)
    y_1 = torch.floor(y)
    x_2 = torch.ceil(x)
    y_2 = torch.ceil(y)

    x_1, x_2 = x_1.long(), x_2.long()
    y_1, y_2 = y_1.long(), y_2.long()

    expend_idx = torch.arange(bs, device=device).repeat_interleave(nums)

    max_idx = feature_map.shape[-1] - 1

    p00 = feature_map[expend_idx, :, x_1.clamp(0, max_idx), y_1.clamp(0, max_idx)]
    p01 = feature_map[expend_idx, :, x_1.clamp(0, max_idx), y_2.clamp(0, max_idx)]
    p10 = feature_map[expend_idx, :, x_2.clamp(0, max_idx), y_1.clamp(0, max_idx)]
    p11 = feature_map[expend_idx, :, x_2.clamp(0, max_idx), y_2.clamp(0, max_idx)]

    p00 = p00.view(bs, -1, c)
    p01 = p01.view(bs, -1, c)
    p10 = p10.view(bs, -1, c)
    p11 = p11.view(bs, -1, c)

    x_1, x_2, y_1, y_2 = x_1.float(), x_2.float(), y_1.float(), y_2.float()

    m = (x_2 - x)
    n = (x - x_1)
    j = (y_2 - y)
    k = (y - y_1)

    feature = (
        m*j).unsqueeze(-1) * p00 + \
        (m*k).unsqueeze(-1) * p10 + \
        (n*j).unsqueeze(-1) * p01 + \
        (n*k).unsqueeze(-1) * p11

    return feature


def ibw_interpolation(feature_map, contour, eps=1e-6):
    device = feature_map.device

    x = contour[:, :, 0].to(device)
    y = contour[:, :, 1].to(device)

    eps = torch.tensor(eps, device=device)

    torch.clamp_(x, min=eps, max=feature_map.shape[-1] - 1)
    torch.clamp_(y, min=eps, max=feature_map.shape[-1] - 1)

    bs, c, _, _ = feature_map.shape
    nums = x.shape[1]

    one = torch.tensor(1.0, device=device)

    x1, x2, x3 = torch.floor(x)-one, torch.floor(x), torch.floor(x)+one
    y1, y2, y3 = torch.floor(y)-one, torch.floor(y), torch.floor(y)+one

    x1, x2, x3 = x1.long(), x2.long(), x3.long()
    y1, y2, y3 = y1.long(), y2.long(), y3.long()

    expend_idx = torch.arange(bs, device=device).repeat_interleave(nums)

    max_idx = feature_map.shape[-1] - 1

    def get_feat(xi, yi):
        return feature_map[expend_idx, :, xi.clamp(0, max_idx), yi.clamp(0, max_idx)].view(bs, -1, c)

    q_lup = get_feat(x1, y1)
    q_up  = get_feat(x1, y2)
    q_rup = get_feat(x1, y3)
    q_l   = get_feat(x2, y1)
    q_c   = get_feat(x2, y2)
    q_r   = get_feat(x2, y3)
    q_lb  = get_feat(x3, y1)
    q_b   = get_feat(x3, y2)
    q_rb  = get_feat(x3, y3)

    x1 = x1.float() + 0.5
    x2 = x2.float() + 0.5
    x3 = x3.float() + 0.5

    y1 = y1.float() + 0.5
    y2 = y2.float() + 0.5
    y3 = y3.float() + 0.5

    dist = lambda a, b: 1.0 / (torch.sqrt((a-x)**2 + (b-y)**2) + eps)

    dist_lup = dist(x1, y1)
    dist_up  = dist(x1, y2)
    dist_rup = dist(x1, y3)
    dist_l   = dist(x2, y1)
    dist_c   = dist(x2, y2)
    dist_r   = dist(x2, y3)
    dist_lb  = dist(x3, y1)
    dist_b   = dist(x3, y2)
    dist_rb  = dist(x3, y3)

    dist_sum = dist_lup + dist_up + dist_rup + dist_l + dist_c + dist_r + dist_lb + dist_b + dist_rb

    feature = (
        dist_lup.unsqueeze(-1)*q_lup +
        dist_up.unsqueeze(-1)*q_up +
        dist_rup.unsqueeze(-1)*q_rup +
        dist_l.unsqueeze(-1)*q_l +
        dist_c.unsqueeze(-1)*q_c +
        dist_r.unsqueeze(-1)*q_r +
        dist_lb.unsqueeze(-1)*q_lb +
        dist_b.unsqueeze(-1)*q_b +
        dist_rb.unsqueeze(-1)*q_rb
    ) / dist_sum.unsqueeze(-1)

    return feature


def get_gcn_feature(cnn_feature, img_poly):
    device = cnn_feature.device
    img_poly = img_poly.to(device)

    img_poly = img_poly.unsqueeze(1)

    feature = torch.nn.functional.grid_sample(
        cnn_feature, img_poly
    ).squeeze(2).permute(0, 2, 1)

    return feature
