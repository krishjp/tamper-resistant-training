import torch

def pgd_attack(model, processor, image, target_label_idx, epsilon=8/255, alpha=2/255, num_iter=20):
    """
    Creates a targeted adversarial example using PGD.
    """
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].clone().detach()
    pixel_values.requires_grad = True
    target_label = torch.tensor([target_label_idx])

    for i in range(num_iter):
        outputs = model(pixel_values)
        loss = torch.nn.functional.cross_entropy(outputs.logits, target_label)

        model.zero_grad()
        loss.backward()
        data_grad = pixel_values.grad.data

        perturbed_image = pixel_values.detach() - alpha * torch.sign(data_grad)

        eta = torch.clamp(perturbed_image - inputs['pixel_values'], min=-epsilon, max=epsilon)
        pixel_values = torch.clamp(inputs['pixel_values'] + eta, min=0, max=1).detach()
        pixel_values.requires_grad = True

    return pixel_values